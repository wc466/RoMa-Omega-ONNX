"""Benchmark matchers: one adapter, several interchangeable backends.

Every variant shares upstream's image loading, resizing, confidence mapping,
match sampling and pixel conversion (``VGGTRoMa.match`` / ``.sample`` /
``.to_pixel_coordinates``). Variants differ only in pipeline scope and in what
computes the network outputs:

  torch_bf16_full    the released model as shipped: "precise" setting
                     (800 coarse / 1280 refine), bidirectional, refiners,
                     bf16 autocast. This is the reference.
  torch_bf16_coarse  coarse matcher only, 512x512, one direction, bf16
  torch_fp32_coarse  same, autocast disabled
  torch_fp16_coarse  same, upstream's autocast forced to IEEE float16
  onnx_fp32_coarse   fp32 encoder.onnx -> matcher.onnx on ONNX Runtime (CPU)
  trt_fp32_coarse    TensorRT engines from the fp32 graphs, TF32 disabled
  trt_fp32_tf32_coarse  same graphs, TensorRT default build (TF32 enabled)
  trt_bf16_coarse    TensorRT engines from the bf16-compute graphs
  trt_fp16_coarse    TensorRT engines from the fp16-compute graphs (the fp16
                     matcher overflows to NaN, in PyTorch too; kept on record)
  trt_fp16enc_bf16mat_coarse  fp16 encoder engine + bf16 matcher engine
  torchtrt_fp32_coarse  PyTorch modules compiled with Torch-TensorRT, fp32,
                     TF32 disabled (needs the separate Torch-TensorRT venv)
  torchtrt_bf16_coarse  Torch-TensorRT bf16 encoder; the bf16 matcher stays in
                     PyTorch (TensorRT has no bf16 ConvTranspose)
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    force_autocast_dtype,
    load_roma_omega,
    no_autocast,
    set_matcher_amp,
)

VARIANTS = (
    "torch_bf16_full",
    "torch_bf16_coarse",
    "torch_fp32_coarse",
    "torch_fp16_coarse",
    "onnx_fp32_coarse",
    "trt_fp32_coarse",
    "trt_fp32_tf32_coarse",
    "trt_bf16_coarse",
    "trt_fp16_coarse",
    "trt_fp16enc_bf16mat_coarse",
    "torchtrt_fp32_coarse",
    "torchtrt_bf16_coarse",
)
TORCHTRT_VARIANTS = {"torchtrt_fp32_coarse": "fp32", "torchtrt_bf16_coarse": "bf16"}
# Graph-backed variants: (encoder precision, matcher precision, file suffix).
GRAPH_VARIANTS = {
    "onnx_fp32_coarse": ("fp32", "fp32", ".onnx"),
    "trt_fp32_coarse": ("fp32", "fp32", ".notf32.engine"),
    "trt_fp32_tf32_coarse": ("fp32", "fp32", ".engine"),
    "trt_bf16_coarse": ("bf16", "bf16", ".engine"),
    "trt_fp16_coarse": ("fp16", "fp16", ".engine"),
    "trt_fp16enc_bf16mat_coarse": ("fp16", "bf16", ".engine"),
}
# Upstream samples 5000 matches. Coarse-only at 512x512 the warp is a 128x128
# grid (16384 cells) and VGGTRoMa.sample draws 4x the count without
# replacement, so the coarse variants use 2000.
DEFAULT_NUM_CORRESP = {"full": 5000, "coarse": 2000}


class _GraphCoarseMatcher(nn.Module):
    """Drop-in for ``VGGTRoMa.matcher`` backed by exported graphs. Returns the
    keys VGGTRoMa.forward reads in stage="matcher". Subclasses implement
    ``_run(img_A, img_B) -> (warp, confidence)`` as numpy arrays.

    The graphs are deterministic, so outputs are cached per input pair: repeat
    runs over the same images with different seeds reuse them.
    """

    input_shape: tuple

    def __init__(self):
        super().__init__()
        self._cache: dict[str, tuple] = {}
        self.seconds: list[float] = []

    def _run(self, img_A: torch.Tensor, img_B: torch.Tensor):
        raise NotImplementedError

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor, bidirectional: bool):
        assert not bidirectional, "matcher graph was exported one-directional"
        assert tuple(img_A.shape) == self.input_shape, (img_A.shape, self.input_shape)
        key = hashlib.sha1(
            img_A.float().cpu().numpy().tobytes() + img_B.float().cpu().numpy().tobytes()
        ).hexdigest()
        if key not in self._cache:
            t0 = time.time()
            self._cache[key] = self._run(img_A, img_B)
            self.seconds.append(time.time() - t0)
        warp, conf = self._cache[key]
        return {
            "warp_AB": torch.from_numpy(warp).to(img_A.device),
            "confidence_AB": torch.from_numpy(conf).to(img_A.device),
            "warp_BA": None,
            "confidence_BA": None,
        }


class OnnxCoarseMatcher(_GraphCoarseMatcher):
    def __init__(self, encoder: Path, matcher: Path, provider: str = "CPUExecutionProvider"):
        super().__init__()
        import onnxruntime as ort

        self.encoder = ort.InferenceSession(str(encoder), providers=[provider])
        self.matcher = ort.InferenceSession(str(matcher), providers=[provider])
        self.feat_names = [i.name for i in self.matcher.get_inputs()]
        assert [o.name for o in self.encoder.get_outputs()] == self.feat_names
        self.input_shape = tuple(self.encoder.get_inputs()[0].shape)

    def _run(self, img_A, img_B):
        a, b = img_A.float().cpu().numpy(), img_B.float().cpu().numpy()
        feats = self.encoder.run(None, {"img_A": a, "img_B": b})
        warp, conf = self.matcher.run(None, dict(zip(self.feat_names, feats)))
        return warp, conf


class TrtCoarseMatcher(_GraphCoarseMatcher):
    def __init__(self, encoder_engine: Path, matcher_engine: Path):
        super().__init__()
        from trt_common import TrtRunner

        self.encoder = TrtRunner(encoder_engine)
        self.matcher = TrtRunner(matcher_engine)
        self.input_shape = self.encoder.shapes["img_A"]

    def _run(self, img_A, img_B):
        feats = self.encoder({"img_A": img_A.float(), "img_B": img_B.float()})
        out = self.matcher(feats)
        return (
            out["warp_AB"].float().cpu().numpy().copy(),
            out["confidence_AB"].float().cpu().numpy().copy(),
        )


class TorchTrtCoarseMatcher(_GraphCoarseMatcher):
    """Encoder and matcher compiled straight from the PyTorch modules with
    Torch-TensorRT (no ONNX). If the matcher cannot be compiled at the requested
    precision it stays a PyTorch module and ``matcher_backend`` says so."""

    def __init__(self, torch_matcher: nn.Module, precision: str, height: int, width: int):
        super().__init__()
        import torch_tensorrt

        from common import EMBED_DIM, PATCH_SIZE, EncoderWrapper, MatcherWrapper, tracing_context

        self.precision = precision
        self._ctx = lambda: tracing_context(precision)
        amp = precision != "fp32"
        encoder = EncoderWrapper(torch_matcher.vggt.aggregator, amp=amp).eval()
        matcher = MatcherWrapper(torch_matcher, amp=amp).eval()
        imgs = tuple(torch.rand(1, 3, height, width, device="cuda") for _ in range(2))
        feats = tuple(
            torch.randn(1, 2, height // PATCH_SIZE, width // PATCH_SIZE, 2 * EMBED_DIM, device="cuda")
            for _ in matcher.layers
        )

        def compile_(module, inputs):
            with self._ctx():
                program = torch.export.export(module, inputs)
                return torch_tensorrt.dynamo.compile(
                    program, inputs=list(inputs), min_block_size=1, disable_tf32=True
                )

        self.encoder = compile_(encoder, imgs)
        try:
            self.matcher = compile_(matcher, feats)
            self.matcher_backend = "torch_tensorrt"
        except Exception as e:  # noqa: BLE001 - recorded, then fall back
            self.matcher = matcher
            self.matcher_backend = f"pytorch ({type(e).__name__} compiling with Torch-TensorRT)"
        self.input_shape = (1, 3, height, width)

    def _run(self, img_A, img_B):
        with torch.inference_mode(), self._ctx():
            feats = self.encoder(img_A.float(), img_B.float())
            warp, conf = self.matcher(*[f.float() for f in feats])
        return warp.float().cpu().numpy().copy(), conf.float().cpu().numpy().copy()


def _image_size(img) -> tuple[int, int]:
    """(width, height) of a path, PIL image or HWC array."""
    if isinstance(img, (str, Path)):
        return Image.open(img).size
    if isinstance(img, Image.Image):
        return img.size
    return img.shape[1], img.shape[0]


class RoMaOmegaVariants:
    """Holds one upstream model and switches it between variants.

    ``select(variant)`` returns an object with the repo's BenchmarkMatcher
    interface (``match`` -> pixel keypoints, ``offset``), whose ``match`` also
    accepts PIL images and HWC uint8 RGB arrays (needed for benchmarks that warp
    images in memory).
    """

    def __init__(
        self,
        artifacts: Path = Path("deployment/artifacts"),
        size: str = "512x512",
        num_corresp: int | None = None,
    ):
        self.model = load_roma_omega()
        self.torch_matcher = self.model.matcher
        self.artifacts, self.size = Path(artifacts), size
        self.num_corresp = num_corresp
        self._graph_matchers: dict[str, _GraphCoarseMatcher] = {}
        self.reduced_count_matches = 0
        self.empty_matches = 0

    def _graph_matcher(self, variant: str) -> _GraphCoarseMatcher:
        if variant not in self._graph_matchers and variant in TORCHTRT_VARIANTS:
            self._graph_matchers[variant] = TorchTrtCoarseMatcher(
                self.torch_matcher, TORCHTRT_VARIANTS[variant], self.model.H_lr, self.model.W_lr
            )
        if variant not in self._graph_matchers:
            enc_prec, mat_prec, suffix = GRAPH_VARIANTS[variant]
            encoder = self.artifacts / f"encoder_{self.size}_{enc_prec}" / f"encoder{suffix}"
            matcher = self.artifacts / f"matcher_{self.size}_{mat_prec}" / f"matcher{suffix}"
            assert encoder.exists() and matcher.exists(), f"{variant}: missing {encoder} or {matcher}"
            cls = OnnxCoarseMatcher if suffix == ".onnx" else TrtCoarseMatcher
            self._graph_matchers[variant] = cls(encoder, matcher)
        return self._graph_matchers[variant]

    def select(self, variant: str):
        assert variant in VARIANTS, variant
        from romaomega.types import BenchmarkMatcher

        model, owner = self.model, self
        scope = "full" if variant.endswith("_full") else "coarse"
        fp32 = variant == "torch_fp32_coarse"
        fp16 = variant == "torch_fp16_coarse"
        num_corresp = self.num_corresp or DEFAULT_NUM_CORRESP[scope]

        # Restore upstream state, then apply this variant.
        model.matcher = self.torch_matcher
        set_matcher_amp(self.torch_matcher, True)
        if scope == "full":
            model.stage = "inference"
            model.apply_setting("precise")
        else:
            model.stage = "matcher"
            model.apply_setting("fast")
            if variant in GRAPH_VARIANTS or variant in TORCHTRT_VARIANTS:
                gm = self._graph_matcher(variant)
                assert gm.input_shape == (1, 3, model.H_lr, model.W_lr), gm.input_shape
                model.matcher = gm
            elif fp32:
                set_matcher_amp(self.torch_matcher, False)

        class Selected(BenchmarkMatcher):
            @property
            def name(self) -> str:
                return f"roma_omega_{variant}"

            @torch.inference_mode()
            def match(self, img_A, img_B):
                if fp32:
                    with no_autocast():
                        preds = model.match(img_A, img_B)
                elif fp16:
                    with force_autocast_dtype(torch.float16):
                        preds = model.match(img_A, img_B)
                else:
                    preds = model.match(img_A, img_B)
                n = num_corresp
                if not model.bidirectional:
                    # Same border mask VGGTRoMa.sample applies before sampling;
                    # sample() draws 4n candidates without replacement.
                    H_A = preds["warp_AB"].shape[1]
                    inside = preds["warp_AB"][0].abs().amax(dim=-1).le(1 - 1 / H_A)
                    usable = int(((preds["overlap_AB"][0, ..., 0] > 0) & inside).sum())
                    if not (torch.isfinite(preds["warp_AB"]).all() and torch.isfinite(preds["overlap_AB"]).all()):
                        # Partly NaN output: sampling from it trips a device-side assert.
                        usable = 0
                    n = min(num_corresp, usable // 4)
                    if n < num_corresp:
                        owner.reduced_count_matches += 1
                    if n == 0:
                        # No usable prediction (e.g. NaN output from an fp16
                        # overflow): report no matches instead of crashing.
                        owner.empty_matches += 1
                        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
                matches = model.sample(preds, n)[0]
                w1, h1 = _image_size(img_A)
                w2, h2 = _image_size(img_B)
                kpts1, kpts2 = model.to_pixel_coordinates(matches, H_A=h1, W_A=w1, H_B=h2, W_B=w2)
                return kpts1.cpu().numpy(), kpts2.cpu().numpy()

            @property
            def offset(self) -> float:
                return 0.5

            @property
            def get_type(self):
                return "dense"

        return Selected()

    def describe(self, variant: str) -> dict:
        m = self.model
        extra = {}
        gm = self._graph_matchers.get(variant)
        if isinstance(gm, TorchTrtCoarseMatcher):
            extra["matcher_backend"] = gm.matcher_backend
        return {
            **extra,
            "stage": m.stage,
            "H_lr": m.H_lr,
            "W_lr": m.W_lr,
            "H_hr": m.H_hr,
            "W_hr": m.W_hr,
            "bidirectional": m.bidirectional,
            "num_corresp": self.num_corresp
            or DEFAULT_NUM_CORRESP["full" if variant.endswith("_full") else "coarse"],
        }
