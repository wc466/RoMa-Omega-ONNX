"""Render DIODE val/outdoor depth maps to PNG for the MINIMA RGB-depth benchmark.

MINIMA's test script expects depth images at ``val/outdoor_depth/...png``; the
official DIODE archive ships depth as ``*_depth.npy`` plus a validity mask. The
MINIMA maintainer points to the DIODE devkit's ``plot_depth_map`` for the
conversion (LSXI7/MINIMA issue 53). This follows that recipe: clip to
[0.5, min(300, 99th percentile)], log, jet colormap up to log(max), invalid
pixels black. Output keeps the native resolution.

Run from the repository root:

    python deployment/prepare_diode.py --data-root data/DIODE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from tqdm import tqdm


def render_depth(dm: np.ndarray, validity_mask: np.ndarray) -> np.ndarray:
    dm = np.squeeze(dm).astype(np.float64)
    valid = np.squeeze(validity_mask) > 0
    min_depth = 0.5
    max_depth = min(300, np.percentile(dm, 99))
    log_dm = np.log(np.clip(dm, min_depth, max_depth))
    # imshow(dm, vmax=log(max_depth)) autoscales vmin to the smallest valid value.
    vmax = np.log(max_depth)
    vmin = log_dm[valid].min() if valid.any() else np.log(min_depth)
    norm = (log_dm - vmin) / max(vmax - vmin, 1e-12)
    rgb = colormaps["jet"](np.clip(norm, 0.0, 1.0))[..., :3]
    rgb[~valid] = 0.0
    return (rgb * 255).round().astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", type=Path, default=Path("data/DIODE"))
    args = p.parse_args()

    src_root = args.data_root / "val" / "outdoor"
    dst_root = args.data_root / "val" / "outdoor_depth"
    rgb_files = sorted(src_root.rglob("*.png"))
    assert rgb_files, f"no images under {src_root}"
    for rgb in tqdm(rgb_files, desc="rendering depth"):
        depth = np.load(rgb.with_name(rgb.stem + "_depth.npy"))
        mask = np.load(rgb.with_name(rgb.stem + "_depth_mask.npy"))
        out = dst_root / rgb.relative_to(src_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(render_depth(depth, mask)).save(out)
    print(f"{len(rgb_files)} depth images -> {dst_root}")


if __name__ == "__main__":
    main()
