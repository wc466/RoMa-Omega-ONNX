"""Shared helpers for the RoMa-Omega deployment scripts.

All scripts in this directory are meant to be run from the repository root,
with the upstream RoMa-Omega code checked out as the ``RoMa-Omega`` submodule
(upstream code resolves ``vggtomega.pt`` and the benchmark data relative to the
working directory). Nothing here modifies upstream model code.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ROOT = REPO_ROOT / "RoMa-Omega"
VGGT_OMEGA_SUBMODULE = UPSTREAM_ROOT / "third_party" / "vggtomega"
# Upstream appends the relative path "third_party/vggtomega" to sys.path, which
# only resolves from the upstream root; add the absolute path so it resolves here.
if str(VGGT_OMEGA_SUBMODULE) not in sys.path:
    sys.path.append(str(VGGT_OMEGA_SUBMODULE))
VGGT_OMEGA_SOURCE = (
    "https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt"
)
VGGT_OMEGA_LICENSE = "FAIR Noncommercial Research License v1"

PATCH_SIZE = 16
EMBED_DIM = 1024
# The released RoMa-Omega matcher reads cached aggregator layers 2 and 3
# (Matcher.Cfg.head_feature_layers, and the deepest one for correlation).
DEFAULT_LAYERS = (2, 3)


class EncoderWrapper(nn.Module):
    """VGGT-Omega aggregator as a pair encoder.

    Mirrors ``romaomega.matcher.Matcher._vggt_features``: stack the two views,
    run the frozen aggregator, drop the camera/register tokens and reshape the
    patch tokens of the selected cached layers to ``(B, 2, H_p, W_p, 2 * D)``.

    Inputs are RGB in [0, 1], shape ``(B, 3, H, W)`` with H and W multiples of
    the patch size. ImageNet normalisation happens inside the aggregator.

    ``amp=True`` reproduces the released model's precision: fp32 weights with
    the aggregator computed under bf16 autocast (as Matcher.forward does).
    Inputs and outputs are fp32 either way.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        amp: bool = False,
    ):
        super().__init__()
        self.aggregator = aggregator
        self.layers = tuple(layers)
        self.patch_size = aggregator.patch_size
        self.amp = amp

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> tuple[torch.Tensor, ...]:
        B = img_A.shape[0]
        H_p = img_A.shape[-2] // self.patch_size
        W_p = img_A.shape[-1] // self.patch_size
        images = torch.stack((img_A, img_B), dim=1)  # (B, 2, 3, H, W)
        with torch.autocast(img_A.device.type, torch.bfloat16, enabled=self.amp):
            tokens_list, patch_token_start = self.aggregator(images)
        cached = [t for t in tokens_list if t is not None]
        return tuple(
            cached[i][:, :, patch_token_start:, :].reshape(B, 2, H_p, W_p, -1).float()
            for i in self.layers
        )


def load_encoder(
    checkpoint: str | Path, layers: tuple[int, ...] = DEFAULT_LAYERS, amp: bool = False
) -> EncoderWrapper:
    """Build the CPU encoder (fp32 weights) from a VGGT-Omega checkpoint (aggregator only)."""
    # Imported lazily: importing romaomega.matcher pulls in the whole package.
    from romaomega.matcher import _load_vggt_omega

    vggt = _load_vggt_omega(str(checkpoint), EMBED_DIM, PATCH_SIZE)
    vggt.eval()
    for p in vggt.parameters():
        p.requires_grad_(False)
    return EncoderWrapper(vggt.aggregator, layers, amp).eval()


@contextlib.contextmanager
def no_autocast():
    """Force fp32 by disabling every ``torch.autocast`` entered inside the block.

    Upstream's DPT head enters ``torch.autocast(..., dtype=torch.bfloat16)``
    unconditionally (romaomega/dpt.py), with no config switch. To export and
    reference a pure fp32 matcher without editing upstream code, ``torch.autocast``
    is temporarily replaced by a subclass that is always disabled.
    """
    original = torch.autocast

    class _DisabledAutocast(original):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            super().__init__(device_type, dtype=dtype, enabled=False, cache_enabled=cache_enabled)

    torch.autocast = _DisabledAutocast
    try:
        yield
    finally:
        torch.autocast = original


@contextlib.contextmanager
def force_autocast_dtype(dtype: torch.dtype):
    """Make every ``torch.autocast`` entered inside the block use ``dtype``.

    Upstream hard-codes bfloat16 in its autocast calls. To export an IEEE fp16
    variant with the same autocast placement, ``torch.autocast`` is temporarily
    replaced by a subclass that overrides the requested dtype.
    """
    original = torch.autocast

    class _ForcedAutocast(original):
        def __init__(self, device_type, dtype_=None, enabled=True, cache_enabled=None, **kw):
            kw.pop("dtype", None)
            super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    torch.autocast = _ForcedAutocast
    try:
        yield
    finally:
        torch.autocast = original


def tracing_context(precision: str):
    """Autocast handling for tracing at a given export precision."""
    if precision == "fp32":
        return no_autocast()
    if precision == "fp16":
        return force_autocast_dtype(torch.float16)
    return contextlib.nullcontext()  # bf16: upstream's autocast as written


def per_offset_local_corr(
    feature0, feature1, warp, local_window, B, K, c, r, h, w, device,
    padding_mode="zeros", sample_mode="bilinear", dtype=torch.float32,
):
    """Same result as ``romaomega.local_correlation.native_torch_local_corr``.

    Upstream samples all K window offsets in one ``grid_sample`` call, which
    materialises a ``(c, h, w * K)`` tensor (3.9 GB at 1280 px for the patch-4
    refiner). Here each offset is sampled separately: the same bilinear
    samples, the same products, the same sum over channels, but K tensors of
    ``(c, h, w)``. Exported runtimes are much faster on this form.
    """
    f0 = feature0 / (c**0.5)
    corr = []
    for k in range(K):
        coords = warp + local_window[:, k][:, None, None]
        sampled = nn.functional.grid_sample(
            feature1, coords, padding_mode=padding_mode, align_corners=False, mode=sample_mode
        )
        corr.append((f0 * sampled).sum(dim=1))
    return torch.stack(corr, dim=1).to(dtype)


@contextlib.contextmanager
def local_corr_impl(name: str):
    """Select the local-correlation implementation used by the refiners:
    ``upstream`` (as written) or ``per_offset`` (per_offset_local_corr).
    Upstream looks the function up as a module global, so it is swapped there
    for the duration of the block; upstream files are not edited."""
    import romaomega.local_correlation as lc

    original = lc.native_torch_local_corr
    if name == "per_offset":
        lc.native_torch_local_corr = per_offset_local_corr
    else:
        assert name == "upstream", name
    try:
        yield
    finally:
        lc.native_torch_local_corr = original


COMPUTE_DTYPE = {
    "fp32": "fp32",
    "bf16": "bf16 (torch.autocast, as upstream)",
    "fp16": "fp16 (upstream's autocast placement, dtype forced to float16)",
}


def load_roma_omega():
    """Build upstream RoMa-Omega (frozen VGGT-Omega + released matcher/refiner
    weights) exactly as ``VGGTRoMa()`` does, without torch.compile."""
    os.environ["COMPILE"] = "0"
    from romaomega.vggtroma import VGGTRoMa

    return VGGTRoMa().eval()


def roma_omega_checkpoint() -> tuple[Path, str]:
    """Local path and source URL of the released matcher/refiner checkpoint."""
    from romaomega.vggtroma import ROMAOMEGA_CHECKPOINT_URL

    path = Path(torch.hub.get_dir()) / "checkpoints" / ROMAOMEGA_CHECKPOINT_URL.split("/")[-1]
    return path, ROMAOMEGA_CHECKPOINT_URL


def set_matcher_amp(matcher: nn.Module, enabled: bool) -> None:
    """Toggle the matcher's own bf16 autocast (Matcher.Cfg.enable_amp) on an
    instance. The Cfg dataclass is frozen, so swap in a modified copy."""
    matcher.cfg = dataclasses.replace(matcher.cfg, enable_amp=enabled)


class MatcherWrapper(nn.Module):
    """RoMa-Omega coarse matcher, from encoder features to warp + confidence.

    Re-states the body of ``romaomega.matcher.Matcher.forward`` after feature
    extraction, calling the matcher's own submodules and helpers, so that the
    encoder can live in a separate graph. Validate against upstream
    ``Matcher.forward`` (see validate_matcher.py), not against this wrapper.

    Inputs are the encoder outputs for ``Matcher.Cfg.head_feature_layers``
    (shallow->deep, the last being the deepest cached layer), each of shape
    ``(B, 2, H_p, W_p, 2 * D)``.

    ``amp=False``: pure fp32; must be traced inside ``no_autocast()`` (the DPT
    head hard-codes autocast). ``amp=True``: the released model's precision,
    with bf16 autocast placed exactly as in Matcher.forward (everything before
    the head) plus the head's own autocast; trace WITHOUT ``no_autocast()``.
    """

    def __init__(self, matcher: nn.Module, bidirectional: bool = False, amp: bool = False):
        super().__init__()
        assert matcher.cfg.backbone == "vggt"
        self.matcher = matcher
        self.amp = amp
        self.layers = tuple(matcher.cfg.head_feature_layers)
        # The deepest cached layer (index 3 of 4) drives correlation.
        assert self.layers[-1] == 3, self.layers
        self.bidirectional = bidirectional

    def forward(self, *feats: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from romaomega.geometry import get_normalized_grid

        m = self.matcher
        from romaomega.device import device

        head_input_BA = None
        with torch.autocast(device.type, torch.bfloat16, enabled=self.amp):
            f_list_A = [f[:, 0] for f in feats]
            f_list_B = [f[:, 1] for f in feats]
            B, H, W, _ = f_list_A[-1].shape
            f_list_A[-1], f_list_B[-1] = m._refine_deepest(f_list_A[-1], f_list_B[-1])
            f_corr_A = m.to_work(f_list_A[-1])
            f_corr_B = m.to_work(f_list_B[-1])
            head_layers_A = [m.to_work(f) for f in f_list_A]
            head_layers_B = [m.to_work(f) for f in f_list_B]

            x = get_normalized_grid(B, H, W)
            x_emb = nn.functional.linear(
                x.reshape(B, H * W, 2), m.scale * m.omega
            ).reshape(B, H, W, -1)
            pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

            _, _, head_input_AB = m._assemble_head_input(
                f_list_src=head_layers_A,
                f_corr_src=f_corr_A,
                f_corr_tgt=f_corr_B,
                pos_emb_grid=pos_emb_grid,
            )
            if self.bidirectional:
                _, _, head_input_BA = m._assemble_head_input(
                    f_list_src=head_layers_B,
                    f_corr_src=f_corr_B,
                    f_corr_tgt=f_corr_A,
                    pos_emb_grid=pos_emb_grid,
                )
        # Head runs outside the matcher autocast, as upstream (fp32 outputs).
        outputs = m._run_head(head_input_AB, None, None)
        if self.bidirectional:
            outputs = outputs + m._run_head(head_input_BA, None, None)
        return outputs


class RefinerStageWrapper(nn.Module):
    """One refinement pass of RoMa-Omega at a fixed image resolution.

    Re-states the body of the refinement loop in ``VGGTRoMa.forward`` for one
    ``(img_A, img_B)`` resolution: VGG19-BN fine features, then the patch-4,
    patch-2 and patch-1 ConvRefiners, each starting from the previous warp and
    confidence resized to its grid. Calls upstream's own submodules and
    ``_interpolate_warp_and_confidence``. Validate against upstream
    ``VGGTRoMa.forward`` (see validate_pipeline.py), not against this wrapper.

    ``second_pass=True`` is the high-resolution pass that follows a first
    refinement pass: upstream then zeroes the incoming precision channels
    before the patch-4 refiner.

    Inputs: images in [0, 1], ``(B, 3, H, W)``; previous warp ``(B, h, w, 2)``
    and confidence ``(B, h, w, C)`` with C = 1 after the matcher and 4 after a
    refinement pass; the same for the B->A direction when ``bidirectional``.
    Outputs: warp ``(B, H, W, 2)`` and confidence ``(B, H, W, 4)`` per direction.
    """

    def __init__(self, model: nn.Module, bidirectional: bool = True, second_pass: bool = False):
        super().__init__()
        assert model.refiners is not None and model.refiner_features is not None
        self.refiners = model.refiners
        self.refiner_features = model.refiner_features
        self.anchor_width = model.anchor_width
        self.anchor_height = model.anchor_height
        self.bidirectional = bidirectional
        self.second_pass = second_pass

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor, *prev: torch.Tensor):
        from romaomega.device import device
        from romaomega.vggtroma import _interpolate_warp_and_confidence

        # Upstream zeroes the precision channels in place; do not mutate inputs.
        warp_AB, confidence_AB = prev[0], prev[1].clone()
        warp_BA = confidence_BA = None
        if self.bidirectional:
            warp_BA, confidence_BA = prev[2], prev[3].clone()
        B, C, H, W = img_A.shape
        scale_factor = torch.tensor(
            (W / self.anchor_width, H / self.anchor_height), device=device
        )
        refiner_features_A = self.refiner_features(img_A)
        refiner_features_B = self.refiner_features(img_B)
        for patch_size_str, refiner in self.refiners.items():
            patch_size = int(patch_size_str)
            zero_out_precision = self.second_pass and patch_size == 4
            warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                warp=warp_AB, confidence=confidence_AB, H=H, W=W,
                patch_size=patch_size, zero_out_precision=zero_out_precision,
            )
            if self.bidirectional:
                warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                    warp=warp_BA, confidence=confidence_BA, H=H, W=W,
                    patch_size=patch_size, zero_out_precision=zero_out_precision,
                )
            f_patch_A = refiner_features_A[patch_size]
            f_patch_B = refiner_features_B[patch_size]
            out_AB = refiner(
                f_A=f_patch_A, f_B=f_patch_B, prev_warp=warp_AB,
                prev_confidence=confidence_AB, scale_factor=scale_factor,
            )
            warp_AB, confidence_AB = out_AB["warp"], out_AB["confidence"]
            if self.bidirectional:
                out_BA = refiner(
                    f_A=f_patch_B, f_B=f_patch_A, prev_warp=warp_BA,
                    prev_confidence=confidence_BA, scale_factor=scale_factor,
                )
                warp_BA, confidence_BA = out_BA["warp"], out_BA["confidence"]
        if self.bidirectional:
            return warp_AB, confidence_AB, warp_BA, confidence_BA
        return warp_AB, confidence_AB


def matcher_output_names(bidirectional: bool) -> list[str]:
    names = ["warp_AB", "confidence_AB"]
    if bidirectional:
        names += ["warp_BA", "confidence_BA"]
    return names


def refiner_io_names(bidirectional: bool) -> tuple[list[str], list[str]]:
    """Input and output tensor names of a refinement-pass graph."""
    dirs = ("AB", "BA") if bidirectional else ("AB",)
    names_in = ["img_A", "img_B"]
    names_out = []
    for d in dirs:
        names_in += [f"prev_warp_{d}", f"prev_confidence_{d}"]
        names_out += [f"warp_{d}", f"confidence_{d}"]
    return names_in, names_out


def output_names(layers: tuple[int, ...]) -> list[str]:
    return [f"feat_{i}" for i in layers]


def parse_layers(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.split(",") if t.strip() != "")


def load_image(path: str | Path, height: int, width: int) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1], shape (1, 3, H, W)."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_state(cwd: Path) -> dict:
    return {
        "commit": _git(["rev-parse", "HEAD"], cwd),
        "remote": _git(["config", "--get", "remote.origin.url"], cwd),
        # Untracked/modified files (e.g. this deployment/ dir on an upstream clone).
        "dirty": _git(["status", "--porcelain"], cwd) != "",
    }


def describe_files(paths: list[Path]) -> dict:
    return {
        p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in paths
    }


def write_provenance(sidecar: Path, record: dict) -> None:
    """Write the provenance sidecar that must accompany every exported graph."""
    import onnx
    import onnxscript

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
        "commits": {
            "deployment": git_state(REPO_ROOT),
            "roma_omega": git_state(UPSTREAM_ROOT),
            "vggt_omega": git_state(VGGT_OMEGA_SUBMODULE),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxscript": onnxscript.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
