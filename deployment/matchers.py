"""RoMa-Omega variants behind the repo's BenchmarkMatcher interface.

One PipelineConfig (pipeline.py: coarse size, refinement size, direction,
refiners on/off) is shared by every variant; a variant chooses what executes
it and at which precision:

  torch_bf16      upstream PyTorch model as released (fp32 weights, bf16 autocast)
  torch_fp32      same, every autocast disabled
  torch_fp16      same, upstream's autocast forced to IEEE float16
  onnx_cpu_fp32   fp32 ONNX graphs on ONNX Runtime, CPU provider
  onnx_gpu_fp32   fp32 ONNX graphs on ONNX Runtime's CUDA provider, TF32
                  disabled (needs the onnxruntime-gpu package)
  trt_fp32        TensorRT engines from the fp32 graphs, TF32 disabled
  trt_fp32_tf32   same graphs, TensorRT default build (TF32 enabled)
  trt_bf16        TensorRT engines from the bf16-compute graphs
  trt_fp16        TensorRT engines from the fp16-compute graphs (the fp16
                  matcher overflows to NaN, in PyTorch too; kept on record)
  trt_fp16enc_bf16  fp16 encoder engine, bf16 engines for everything after it
  torchtrt_fp32   PyTorch modules compiled with Torch-TensorRT, fp32, TF32
                  disabled (needs the separate Torch-TensorRT environment)
  torchtrt_bf16   same in bf16; modules TensorRT cannot compile stay in PyTorch
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    force_autocast_dtype,
    load_roma_omega,
    no_autocast,
    refiner_io_names,
    set_matcher_amp,
    tracing_context,
)
from pipeline import PipelineConfig  # noqa: E402

TORCH_VARIANTS = {"torch_bf16": "bf16", "torch_fp32": "fp32", "torch_fp16": "fp16"}
# ONNX Runtime execution provider per variant. The CUDA provider enables TF32
# by default; it is disabled here to match the TensorRT --no-tf32 engines.
ONNX_VARIANTS = {
    "onnx_cpu_fp32": "CPUExecutionProvider",
    "onnx_gpu_fp32": ("CUDAExecutionProvider", {"use_tf32": "0"}),
}
# TensorRT variants: (encoder precision, precision of the other graphs, engine suffix).
TRT_VARIANTS = {
    "trt_fp32": ("fp32", "fp32", ".notf32.engine"),
    "trt_fp32_tf32": ("fp32", "fp32", ".engine"),
    "trt_bf16": ("bf16", "bf16", ".engine"),
    "trt_fp16": ("fp16", "fp16", ".engine"),
    "trt_fp16enc_bf16": ("fp16", "bf16", ".engine"),
}
TORCHTRT_VARIANTS = {"torchtrt_fp32": "fp32", "torchtrt_bf16": "bf16"}
VARIANTS = (*TORCH_VARIANTS, *ONNX_VARIANTS, *TRT_VARIANTS, *TORCHTRT_VARIANTS)
# Upstream samples 5000 matches per pair.
DEFAULT_NUM_CORRESP = 5000
# Graph outputs are cached per image pair (repeat seeds reuse them). A refined
# 1280 px bidirectional prediction is ~80 MB, so the cache is bounded.
CACHE_PAIRS = 160


def engine_path(onnx_path: Path, suffix: str) -> Path:
    return onnx_path.with_name(onnx_path.stem + suffix)


class OnnxRunner:
    """One ONNX graph on ONNX Runtime; torch CUDA tensors in and out."""

    def __init__(self, path: Path, provider="CPUExecutionProvider"):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=[provider])
        # ONNX Runtime falls back to CPU silently if a provider fails to load.
        wanted = provider if isinstance(provider, str) else provider[0]
        assert self.sess.get_providers()[0] == wanted, self.sess.get_providers()
        self.gpu = wanted != "CPUExecutionProvider"
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.shapes = {i.name: tuple(i.shape) for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]
        if self.gpu:
            # Bind torch tensors directly so nothing is copied through the host.
            self.outputs = {
                o.name: torch.empty(tuple(o.shape), dtype=torch.float32, device="cuda")
                for o in self.sess.get_outputs()
            }

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.gpu:
            feed = {n: inputs[n].float().cpu().numpy() for n in self.input_names}
            out = self.sess.run(None, feed)
            return {n: torch.from_numpy(o).cuda() for n, o in zip(self.output_names, out)}
        binding = self.sess.io_binding()
        held = []
        for n in self.input_names:
            t = inputs[n].to(device="cuda", dtype=torch.float32).contiguous()
            assert tuple(t.shape) == self.shapes[n], (n, t.shape, self.shapes[n])
            held.append(t)
            binding.bind_input(n, "cuda", 0, np.float32, tuple(t.shape), t.data_ptr())
        for n, t in self.outputs.items():
            binding.bind_output(n, "cuda", 0, np.float32, tuple(t.shape), t.data_ptr())
        torch.cuda.synchronize()  # inputs are ready
        self.sess.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return self.outputs


class TorchRunner:
    """A PyTorch (or Torch-TensorRT compiled) module with named inputs/outputs."""

    def __init__(self, module, input_names, output_names, precision: str):
        self.module, self.precision = module, precision
        self.input_names, self.output_names = list(input_names), list(output_names)

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.inference_mode(), tracing_context(self.precision):
            out = self.module(*[inputs[n].float() for n in self.input_names])
        out = out if isinstance(out, (tuple, list)) else (out,)
        return {n: o.float() for n, o in zip(self.output_names, out)}


class GraphPipeline:
    """Chain of per-stage runners with the call signature of ``VGGTRoMa.forward``.

    ``runners`` maps each stage of ``cfg.stages`` to a callable taking and
    returning a dict of named CUDA tensors. Returns the keys ``VGGTRoMa.match``
    reads: warp/confidence for A->B, and for B->A (None when one-directional).
    """

    def __init__(self, cfg: PipelineConfig, runners: dict, backends: dict | None = None):
        assert tuple(runners) == cfg.stages, (tuple(runners), cfg.stages)
        self.cfg, self.runners = cfg, runners
        self.backends = backends or {}
        self.seconds: dict[str, list[float]] = {s: [] for s in cfg.stages}
        self.timing = False
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.use_cache = True

    def _run_stage(self, stage: str, inputs: dict) -> dict:
        if not self.timing:
            return self.runners[stage](inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.runners[stage](inputs)
        torch.cuda.synchronize()
        self.seconds[stage].append((time.perf_counter() - t0) * 1e3)
        return out

    def run(self, img_A_lr, img_B_lr, img_A_hr=None, img_B_hr=None) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        assert tuple(img_A_lr.shape[-2:]) == cfg.coarse, (img_A_lr.shape, cfg.coarse)
        feats = self._run_stage("encoder", {"img_A": img_A_lr, "img_B": img_B_lr})
        out = self._run_stage("matcher", feats)
        if cfg.refiners:
            out = self._run_stage(
                "refiner",
                {"img_A": img_A_lr, "img_B": img_B_lr, **{f"prev_{k}": v for k, v in out.items()}},
            )
            if cfg.refine is not None:
                assert tuple(img_A_hr.shape[-2:]) == cfg.refine, (img_A_hr.shape, cfg.refine)
                out = self._run_stage(
                    "refiner2",
                    {"img_A": img_A_hr, "img_B": img_B_hr, **{f"prev_{k}": v for k, v in out.items()}},
                )
        return out

    def forward(self, img_A_lr, img_B_lr, img_A_hr=None, img_B_hr=None):
        """Drop-in for ``VGGTRoMa.forward`` in eval mode."""
        key = None
        if self.use_cache:
            key = hashlib.sha1(
                img_A_lr.float().cpu().numpy().tobytes() + img_B_lr.float().cpu().numpy().tobytes()
            ).hexdigest()
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            out = {k: v.to(img_A_lr.device) for k, v in self._cache[key].items()}
        else:
            out = self.run(img_A_lr, img_B_lr, img_A_hr, img_B_hr)
            # Runners reuse their output buffers: copy what is kept.
            out = {k: v.float().clone() for k, v in out.items()}
            if key is not None:
                self._cache[key] = {k: v.cpu() for k, v in out.items()}
                while len(self._cache) > CACHE_PAIRS:
                    self._cache.popitem(last=False)
        return {
            "warp_AB": out["warp_AB"],
            "confidence_AB": out["confidence_AB"],
            "warp_BA": out.get("warp_BA"),
            "confidence_BA": out.get("confidence_BA"),
        }


def build_onnx_pipeline(cfg: PipelineConfig, variant: str, artifacts: Path) -> GraphPipeline:
    runners = {}
    for stage in cfg.stages:
        path = cfg.graph_path(stage, "fp32", artifacts)
        assert path.exists(), f"{variant}: missing {path}"
        runners[stage] = OnnxRunner(path, ONNX_VARIANTS[variant])
    return GraphPipeline(cfg, runners, {s: "onnxruntime" for s in cfg.stages})


def trt_engine_paths(cfg: PipelineConfig, variant: str, artifacts: Path) -> dict[str, Path]:
    enc_prec, rest_prec, suffix = TRT_VARIANTS[variant]
    return {
        stage: engine_path(
            cfg.graph_path(stage, enc_prec if stage == "encoder" else rest_prec, artifacts), suffix
        )
        for stage in cfg.stages
    }


def build_trt_pipeline(cfg: PipelineConfig, variant: str, artifacts: Path) -> GraphPipeline:
    from trt_common import TrtRunner

    runners = {}
    for stage, path in trt_engine_paths(cfg, variant, artifacts).items():
        assert path.exists(), f"{variant}: missing {path}"
        runners[stage] = TrtRunner(path)
    return GraphPipeline(cfg, runners, {s: "tensorrt" for s in cfg.stages})


def build_torchtrt_pipeline(cfg: PipelineConfig, variant: str, model) -> GraphPipeline:
    """Compile each stage straight from the PyTorch modules with Torch-TensorRT
    (no ONNX). A stage that cannot be compiled stays a PyTorch module; the
    returned pipeline's ``backends`` says which."""
    import torch_tensorrt

    from common import (
        EMBED_DIM,
        PATCH_SIZE,
        EncoderWrapper,
        MatcherWrapper,
        RefinerStageWrapper,
        matcher_output_names,
        output_names,
    )

    precision = TORCHTRT_VARIANTS[variant]
    amp = precision != "fp32"
    (ch, cw), bidir = cfg.coarse, cfg.bidirectional
    n_dir = 2 if bidir else 1
    img = lambda h, w: torch.rand(1, 3, h, w, device="cuda")  # noqa: E731
    matcher = MatcherWrapper(model.matcher, bidirectional=bidir, amp=amp).eval()
    feat_names = output_names(matcher.layers)
    ref_in, ref_out = refiner_io_names(bidir)

    def prev(h, w, c):
        out = []
        for _ in range(n_dir):
            out += [torch.rand(1, h, w, 2, device="cuda") * 2 - 1, torch.randn(1, h, w, c, device="cuda")]
        return out

    stages = {
        "encoder": (
            EncoderWrapper(model.matcher.vggt.aggregator, amp=amp).eval(),
            (img(ch, cw), img(ch, cw)), ["img_A", "img_B"], feat_names,
        ),
        "matcher": (
            matcher,
            tuple(torch.randn(1, 2, ch // PATCH_SIZE, cw // PATCH_SIZE, 2 * EMBED_DIM, device="cuda")
                  for _ in feat_names),
            feat_names, matcher_output_names(bidir),
        ),
    }
    if cfg.refiners:
        stages["refiner"] = (
            RefinerStageWrapper(model, bidir, second_pass=False).eval(),
            (img(ch, cw), img(ch, cw), *prev(ch // 4, cw // 4, 1)), ref_in, ref_out,
        )
        if cfg.refine is not None:
            rh, rw = cfg.refine
            stages["refiner2"] = (
                RefinerStageWrapper(model, bidir, second_pass=True).eval(),
                (img(rh, rw), img(rh, rw), *prev(ch, cw, 4)), ref_in, ref_out,
            )

    runners, backends = {}, {}
    for stage, (module, example, names_in, names_out) in stages.items():
        try:
            with tracing_context(precision):
                program = torch.export.export(module, tuple(example))
                compiled = torch_tensorrt.dynamo.compile(
                    program, inputs=list(example), min_block_size=1, disable_tf32=True
                )
            backends[stage] = "torch_tensorrt"
        except Exception as e:  # noqa: BLE001 - recorded, then fall back
            compiled = module
            backends[stage] = f"pytorch ({type(e).__name__} compiling with Torch-TensorRT)"
        runners[stage] = TorchRunner(compiled, names_in, names_out, precision)
    return GraphPipeline(cfg, runners, backends)


def _image_size(img) -> tuple[int, int]:
    """(width, height) of a path, PIL image or HWC array."""
    if isinstance(img, (str, Path)):
        return Image.open(img).size
    if isinstance(img, Image.Image):
        return img.size
    return img.shape[1], img.shape[0]


def apply_config(model, cfg: PipelineConfig) -> None:
    """Put an upstream VGGTRoMa instance into ``cfg`` (what apply_setting does
    for the named presets)."""
    model.stage = "inference" if cfg.refiners else "matcher"
    model.H_lr, model.W_lr = cfg.coarse
    model.H_hr, model.W_hr = cfg.refine if cfg.refine is not None else (None, None)
    model.bidirectional = cfg.bidirectional
    model.threshold = None
    model.balanced_sampling = True


class RoMaOmegaVariants:
    """Holds one upstream model and switches it between variants.

    ``select(variant)`` returns an object with the repo's BenchmarkMatcher
    interface (``match`` -> pixel keypoints, ``offset``), whose ``match`` also
    accepts PIL images and HWC uint8 RGB arrays (needed for benchmarks that warp
    images in memory). Image loading, resizing, confidence mapping and match
    sampling are always upstream's own code; only ``forward`` is swapped.
    """

    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        artifacts: Path = Path("deployment/artifacts"),
        num_corresp: int | None = None,
    ):
        self.cfg = cfg or PipelineConfig()
        self.model = load_roma_omega()
        self.artifacts = Path(artifacts)
        self.num_corresp = num_corresp or DEFAULT_NUM_CORRESP
        self._pipelines: dict[str, GraphPipeline] = {}
        self.reduced_count_matches = 0
        self.empty_matches = 0

    def pipeline(self, variant: str) -> GraphPipeline:
        if variant not in self._pipelines:
            if variant in ONNX_VARIANTS:
                p = build_onnx_pipeline(self.cfg, variant, self.artifacts)
            elif variant in TRT_VARIANTS:
                p = build_trt_pipeline(self.cfg, variant, self.artifacts)
            else:
                p = build_torchtrt_pipeline(self.cfg, variant, self.model)
            self._pipelines[variant] = p
        return self._pipelines[variant]

    def release(self, variant: str) -> None:
        """Drop a graph pipeline (sessions, engines, cached outputs)."""
        if self._pipelines.pop(variant, None) is not None:
            self.model.__dict__.pop("forward", None)
            torch.cuda.empty_cache()

    def select(self, variant: str):
        assert variant in VARIANTS, variant
        from romaomega.types import BenchmarkMatcher

        model, owner = self.model, self
        precision = TORCH_VARIANTS.get(variant)
        num_corresp = self.num_corresp

        # Restore upstream state, then apply this variant.
        model.__dict__.pop("forward", None)
        set_matcher_amp(model.matcher, precision != "fp32")
        apply_config(model, self.cfg)
        if precision is None:
            # Instance attribute shadows VGGTRoMa.forward; upstream is not edited.
            model.forward = self.pipeline(variant).forward

        class Selected(BenchmarkMatcher):
            @property
            def name(self) -> str:
                return f"roma_omega_{variant}"

            @torch.inference_mode()
            def match(self, img_A, img_B):
                if precision == "fp32":
                    with no_autocast():
                        preds = model.match(img_A, img_B)
                elif precision == "fp16":
                    with force_autocast_dtype(torch.float16):
                        preds = model.match(img_A, img_B)
                else:
                    preds = model.match(img_A, img_B)
                # Same border mask VGGTRoMa.sample applies before sampling;
                # sample() draws 4n candidates without replacement.
                H_A = preds["warp_AB"].shape[1]
                usable, finite = 0, True
                for d in owner.cfg.directions:
                    warp, overlap = preds[f"warp_{d}"], preds[f"overlap_{d}"]
                    inside = warp[0].abs().amax(dim=-1).le(1 - 1 / H_A)
                    usable += int(((overlap[0, ..., 0] > 0) & inside).sum())
                    finite &= bool(torch.isfinite(warp).all() and torch.isfinite(overlap).all())
                    if "precision_" + d in preds:
                        finite &= bool(torch.isfinite(preds["precision_" + d]).all())
                if not finite:
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
        p = self._pipelines.get(variant)
        return {
            **self.cfg.describe(),
            "num_corresp": self.num_corresp,
            "backends": p.backends if p is not None else {s: "pytorch" for s in self.cfg.stages},
        }
