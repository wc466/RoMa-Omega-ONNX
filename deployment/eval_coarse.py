"""Coarse-only sparse benchmark: ONNX graphs vs PyTorch, across seeds.

Scores the coarse matcher alone (no refiners) through the repo's own sparse
benchmarks, for three variants that differ only in what computes the warp:

  * onnx_fp32   - encoder.onnx -> matcher.onnx on ONNX Runtime
  * torch_fp32  - upstream Matcher.forward with autocast disabled
  * torch_bf16  - upstream Matcher.forward as shipped (bf16 autocast)

Everything else is upstream and shared: VGGTRoMa.match() does image loading,
bicubic resize and confidence mapping (stage="matcher", setting "fast" =
512x512, one direction), and romaomega.matchers.RoMaOmegaMatcher samples 5000
matches and converts them to pixels. Each variant is run once per seed; the
seed affects match sampling and MAGSAC, not the network.

Run from the repository root:

    python deployment/eval_coarse.py \
        --encoder deployment/artifacts/encoder_512x512_fp32/encoder.onnx \
        --matcher deployment/artifacts/matcher_512x512_fp32/matcher.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import onnxruntime as ort
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_roma_omega, no_autocast, set_matcher_amp  # noqa: E402

VARIANTS = ("onnx_fp32", "torch_fp32", "torch_bf16")


class OnnxCoarseMatcher(nn.Module):
    """Drop-in for ``VGGTRoMa.matcher`` that runs the exported encoder and
    matcher graphs. Returns the keys VGGTRoMa.forward reads in stage="matcher".

    The graphs are deterministic, so outputs are cached per input pair: repeat
    runs over the same images with different seeds reuse them.
    """

    def __init__(self, encoder: Path, matcher: Path, provider: str):
        super().__init__()
        self.encoder = ort.InferenceSession(str(encoder), providers=[provider])
        self.matcher = ort.InferenceSession(str(matcher), providers=[provider])
        self.feat_names = [i.name for i in self.matcher.get_inputs()]
        assert [o.name for o in self.encoder.get_outputs()] == self.feat_names
        self.input_shape = tuple(self.encoder.get_inputs()[0].shape)
        self._cache: dict[str, tuple] = {}
        self.seconds: list[float] = []

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor, bidirectional: bool):
        assert not bidirectional, "matcher graph was exported one-directional"
        assert tuple(img_A.shape) == self.input_shape, (img_A.shape, self.input_shape)
        a = img_A.float().cpu().numpy()
        b = img_B.float().cpu().numpy()
        key = hashlib.sha1(a.tobytes() + b.tobytes()).hexdigest()
        if key not in self._cache:
            t0 = time.time()
            feats = self.encoder.run(None, {"img_A": a, "img_B": b})
            warp, conf = self.matcher.run(None, dict(zip(self.feat_names, feats)))
            self.seconds.append(time.time() - t0)
            self._cache[key] = (warp, conf)
        warp, conf = self._cache[key]
        return {
            "warp_AB": torch.from_numpy(warp).to(img_A.device),
            "confidence_AB": torch.from_numpy(conf).to(img_A.device),
            "warp_BA": None,
            "confidence_BA": None,
        }


def make_bench_matcher(model, num_corresp: int):
    """Benchmark adapter mirroring ``romaomega.matchers.RoMaOmegaMatcher`` with
    a configurable number of matches.

    Upstream hard-codes 5000, and ``VGGTRoMa.sample`` draws 4x that many
    candidates without replacement. Coarse-only at 512x512 the warp is a
    128x128 grid (16384 cells), so 5000 cannot be sampled. If a pair has too
    few cells with non-zero overlap, the count is reduced for that pair.
    """
    from PIL import Image

    from romaomega.types import BenchmarkMatcher

    class CoarseMatcher(BenchmarkMatcher):
        reduced_pairs = 0

        @property
        def name(self) -> str:
            return "roma_omega_coarse"

        @torch.inference_mode()
        def match(self, img_A_path: str, img_B_path: str):
            preds = model.match(img_A_path, img_B_path)
            # Same border mask VGGTRoMa.sample applies before sampling.
            H_A = preds["warp_AB"].shape[1]
            inside = preds["warp_AB"][0].abs().amax(dim=-1).le(1 - 1 / H_A)
            usable = int(((preds["overlap_AB"][0, ..., 0] > 0) & inside).sum())
            n = min(num_corresp, usable // 4)
            if n < num_corresp:
                CoarseMatcher.reduced_pairs += 1
            matches = model.sample(preds, n)[0]

            w1, h1 = Image.open(img_A_path).size
            w2, h2 = Image.open(img_B_path).size
            kpts1, kpts2 = model.to_pixel_coordinates(matches, H_A=h1, W_A=w1, H_B=h2, W_B=w2)
            return kpts1.cpu().numpy(), kpts2.cpu().numpy()

        @property
        def offset(self) -> float:
            return 0.5

        @property
        def get_type(self):
            return "dense"

    return CoarseMatcher()


def build_benchmark(name: str):
    if name == "wxbs":
        from romaomega.benchmarks.general import WxBSBenchmark

        return WxBSBenchmark()
    if name == "hardmatch":
        from hardmatch import HardMatchBenchmark

        return HardMatchBenchmark()
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--matcher", type=Path, required=True)
    p.add_argument("--benchmark", choices=["wxbs", "hardmatch"], default="wxbs")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--num-corresp", type=int, default=2000, help="matches sampled per pair")
    p.add_argument("--provider", default="CPUExecutionProvider")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    variants = [v for v in args.variants.split(",") if v]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    assert all(v in VARIANTS for v in variants), variants

    from romaomega.random import set_seed

    Path("data").mkdir(exist_ok=True)
    model = load_roma_omega()
    # Coarse matcher only, at the resolution the graphs were exported for.
    model.stage = "matcher"
    model.apply_setting("fast")
    torch_matcher = model.matcher
    onnx_matcher = None
    if "onnx_fp32" in variants:
        onnx_matcher = OnnxCoarseMatcher(args.encoder, args.matcher, args.provider)
        assert onnx_matcher.input_shape == (1, 3, model.H_lr, model.W_lr)

    bench_matcher = make_bench_matcher(model, args.num_corresp)
    benchmark = build_benchmark(args.benchmark)

    def run_once(variant: str, seed: int) -> dict:
        set_seed(seed)
        if variant == "onnx_fp32":
            model.matcher = onnx_matcher
            return benchmark.benchmark(bench_matcher)
        model.matcher = torch_matcher
        if variant == "torch_bf16":
            set_matcher_amp(torch_matcher, True)
            return benchmark.benchmark(bench_matcher)
        set_matcher_amp(torch_matcher, False)
        try:
            with no_autocast():
                return benchmark.benchmark(bench_matcher)
        finally:
            set_matcher_amp(torch_matcher, True)

    report = {
        "benchmark": args.benchmark,
        "setting": {"stage": "matcher", "resolution": [model.H_lr, model.W_lr], "bidirectional": False},
        "encoder": str(args.encoder),
        "matcher": str(args.matcher),
        "provider": args.provider,
        "seeds": seeds,
        "num_corresp": args.num_corresp,
        "variants": {},
    }
    for variant in variants:
        runs = []
        for seed in seeds:
            t0 = time.time()
            res = run_once(variant, seed)
            runs.append(
                {"seed": seed, "mAA_10px": res["mAA_10px"], "seconds": round(time.time() - t0, 1)}
            )
            print(f"== {variant} seed {seed}: mAA_10px {res['mAA_10px']:.4f}", flush=True)
        scores = [r["mAA_10px"] for r in runs]
        report["variants"][variant] = {
            "runs": runs,
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }
    model.matcher = torch_matcher
    # Pair evaluations (over all variants and seeds) that had to sample fewer matches.
    report["reduced_count_pair_runs"] = type(bench_matcher).reduced_pairs
    if onnx_matcher is not None and onnx_matcher.seconds:
        report["onnx_seconds_per_pair_mean"] = statistics.mean(onnx_matcher.seconds)

    out = args.out or Path(
        f"deployment/results/coarse_{args.benchmark}_{model.H_lr}x{model.W_lr}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{args.benchmark} mAA@10px, coarse only, {model.H_lr}x{model.W_lr}, seeds {seeds}")
    for variant, v in report["variants"].items():
        print(
            f"  {variant:11s} mean={v['mean']:.4f} std={v['std']:.4f} "
            f"min={v['min']:.4f} max={v['max']:.4f}"
        )
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
