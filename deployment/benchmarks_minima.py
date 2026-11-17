"""Homography benchmarks following the MINIMA evaluation protocol.

Ported from https://github.com/LSXI7/MINIMA (Apache-2.0):
``test_relative_homo_depth.py`` (DIODE, RGB-depth) and
``test_relative_homo_mmim.py`` (MMIM, medical and remote sensing).

MINIMA's scripts do not run in this environment (``collections.Mapping`` and
``np.trapz`` were removed from current Python/NumPy), so the small metric
functions are re-stated here. The random homography sampler is imported from a
MINIMA checkout so the warps follow their code exactly.

Protocol, per pair: match -> ``cv2.findHomography(RANSAC)`` -> mean distance
between the four image corners mapped by the estimated and the ground-truth
homography -> AUC of that error at thresholds 1..20 px, per scene.

Known deviations from MINIMA (see README):
  * DIODE depth PNGs are rendered by prepare_diode.py from the raw .npy depth
    with the DIODE devkit recipe; MINIMA's own rendering is not published.
  * DIODE pairs are enumerated from disk, not read from MINIMA's csv list.
"""

from __future__ import annotations

import collections
import collections.abc
import json
import sys
from math import pi
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

AUC_THRESHOLDS = [1, 3, 5, 7, 10, 15, 20]
MMIM_GROUPS = {
    "mmim_medical": "test_list.txt",
    "mmim_remote": "test_list_2.txt",
}
MMIM_DIRNAME = "Multi-modality-image-matching-database-metrics-methods"


def _import_sample_homography(minima_root: Path):
    # MINIMA's sampler uses collections.Mapping (removed in Python 3.10).
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping
    sys.path.insert(0, str(minima_root))
    try:
        from src.utils.sample_h import sample_homography
    finally:
        sys.path.remove(str(minima_root))
    return sample_homography


def order_corners(corners: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect


def mean_corner_distance(real_H: np.ndarray, pred_H: np.ndarray, H: int, W: int) -> float:
    """MINIMA ``compute_mean_distance`` without the visualisation."""
    corners = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]])
    real_H = np.linalg.inv(real_H)
    pred_H = np.linalg.inv(pred_H)
    real_warped = np.dot(corners, np.transpose(real_H))
    real_warped = real_warped[:, :2] / real_warped[:, 2:]
    warped = np.dot(corners, np.transpose(pred_H))
    warped = warped[:, :2] / warped[:, 2:]
    return float(np.mean(np.linalg.norm(order_corners(real_warped) - order_corners(warped), axis=1)))


def error_auc(errors, thresholds=AUC_THRESHOLDS) -> dict:
    """MINIMA ``error_auc`` (np.trapz -> np.trapezoid)."""
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs[f"auc@{thr}"] = float(np.trapezoid(y, x) / thr)
    return aucs


class HomographyBenchmark:
    """Scenes of pairs with a ground-truth homography from image 0 to image 1."""

    name: str
    scenes: dict[str, list[dict]]

    def load(self, pair: dict):
        """Return (img0, img1) as paths or HWC uint8 RGB arrays."""
        return pair["im0"], pair["im1"]

    def size(self, pair: dict, img0) -> tuple[int, int]:
        """(H, W) of image 0, the frame the corner error is measured in."""
        if isinstance(img0, np.ndarray):
            return img0.shape[0], img0.shape[1]
        from PIL import Image

        w, h = Image.open(img0).size
        return h, w

    def benchmark(self, model) -> dict:
        per_scene = {}
        for scene, pairs in self.scenes.items():
            errors, failed = [], 0
            for pair in tqdm(pairs, desc=f"{self.name} {scene}"):
                img0, img1 = self.load(pair)
                H, W = self.size(pair, img0)
                kpts0, kpts1 = model.match(img0, img1)
                kpts0 = kpts0 - model.offset
                kpts1 = kpts1 - model.offset
                pred_H = None
                if len(kpts0) >= 4:
                    pred_H, _ = cv2.findHomography(kpts0, kpts1, cv2.RANSAC)
                if pred_H is None:
                    errors.append(np.inf)
                    failed += 1
                    continue
                try:
                    errors.append(mean_corner_distance(pair["H"], pred_H, H, W))
                except np.linalg.LinAlgError:
                    errors.append(np.inf)
                    failed += 1
            finite = [e for e in errors if np.isfinite(e)]
            per_scene[scene] = {
                **{k: 100 * v for k, v in error_auc(errors).items()},
                "pairs": len(pairs),
                "failed": failed,
                "median_error_px": float(np.median(finite)) if finite else None,
            }
        keys = [f"auc@{t}" for t in AUC_THRESHOLDS]
        mean = {k: float(np.mean([s[k] for s in per_scene.values()])) for k in keys}
        return {
            "scenes": per_scene,
            "mean_over_scenes": mean,
            "pairs": int(sum(s["pairs"] for s in per_scene.values())),
            "failed": int(sum(s["failed"] for s in per_scene.values())),
        }


class DiodeBenchmark(HomographyBenchmark):
    """RGB-depth pairs from DIODE val/outdoor, each view warped by a random
    homography (MINIMA config), at 640x480, in both directions."""

    name = "diode"
    # Path prefix MINIMA joins onto each listed file; the homography of a pair
    # is seeded from this string, so it is kept regardless of where data lives.
    SEED_ROOT = "./data/DIODE/"
    H_CONFIG = {
        "perspective": True,
        "scaling": True,
        "rotation": True,
        "translation": True,
        "n_scales": 5,
        "n_angles": 25,
        "scaling_amplitude": 0.05,
        "perspective_amplitude_x": 0.05,
        "perspective_amplitude_y": 0.05,
        "patch_ratio": 0.8,
        "max_angle": 10 * (pi / 180),
        "allow_artifacts": False,
        "translation_overflow": 0.0,
    }

    def __init__(self, data_root: Path, minima_root: Path, max_pairs: int | None = None):
        import os

        sample_homography = _import_sample_homography(minima_root)
        data_root = Path(data_root)
        rgb_files = sorted((data_root / "val" / "outdoor").rglob("*.png"))
        assert rgb_files, f"no DIODE images under {data_root}/val/outdoor"
        pairs = []
        for rgb in rgb_files:
            rel = rgb.relative_to(data_root).as_posix()  # val/outdoor/scene/scan/name.png
            depth = data_root / rel.replace("val/outdoor/", "val/outdoor_depth/")
            assert depth.exists(), f"{depth} missing; run deployment/prepare_diode.py"
            seed_rgb = os.path.join(self.SEED_ROOT, rel)
            seed_depth = seed_rgb.replace("/outdoor/", "/outdoor_depth/")
            H0 = sample_homography([480, 640], seed_rgb, dict(self.H_CONFIG)).squeeze().numpy()
            H1 = sample_homography([480, 640], seed_depth, dict(self.H_CONFIG)).squeeze().numpy()
            pairs.append({"im0": str(rgb), "im1": str(depth), "H": H0})
            pairs.append({"im0": str(depth), "im1": str(rgb), "H": H1})
        if max_pairs:
            pairs = pairs[:max_pairs]
        self.scenes = {"DIO-depth": pairs}

    @staticmethod
    def _read(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.resize(img, (640, 480))

    def load(self, pair: dict):
        from kornia.geometry.transform import warp_perspective

        im0 = self._read(pair["im0"])
        im1 = self._read(pair["im1"])
        t1 = torch.tensor(im1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        H = torch.tensor(pair["H"], dtype=torch.float32).unsqueeze(0)
        t1 = warp_perspective(t1, H, t1.shape[2:], align_corners=True)
        im1 = (t1.squeeze(0).permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        # OpenCV reads BGR; the model expects RGB.
        return (
            np.ascontiguousarray(im0[:, :, ::-1]),
            np.ascontiguousarray(im1[:, :, ::-1]),
        )


class MmimBenchmark(HomographyBenchmark):
    """MMIM medical or remote-sensing pairs with ground-truth transforms."""

    def __init__(self, data_root: Path, group: str):
        import scipy.io as sio

        self.name = group
        root = Path(data_root) / MMIM_DIRNAME
        names = [n.strip() for n in (root / MMIM_GROUPS[group]).read_text().splitlines() if n.strip()]
        shift = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]])
        self.scenes = {}
        for name in names:
            scene_dir = root / "Multimodal_Image_Matching_Datasets" / name
            pairs = []
            for files in json.loads((scene_dir / "list.json").read_text()).values():
                T = sio.loadmat(str(scene_dir / files[0]))["T"].T
                T = T / T[2, 2]
                # MATLAB 1-based -> 0-based pixel coordinates.
                T = shift @ T @ np.linalg.inv(shift)
                T = T / T[2, 2]
                pairs.append(
                    {"im0": str(scene_dir / files[2]), "im1": str(scene_dir / files[1]), "H": T}
                )
            self.scenes[name.strip("/")] = pairs
