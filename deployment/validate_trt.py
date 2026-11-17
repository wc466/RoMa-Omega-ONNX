"""Validate and time the TensorRT encoder and matcher engines.

Accuracy: the engine chain (encoder.engine -> matcher.engine) is compared with
ONNX Runtime CPU on the same graphs, and with upstream ``Matcher.forward`` in
fp32 and in bf16 autocast (as shipped).

Latency: median over repeated runs after warm-up, GPU-synchronised, for the
engines and for upstream PyTorch on the same inputs. Coarse matcher only, no
refiners, no image loading or match sampling.

Run from the repository root:

    python deployment/validate_trt.py \
        --encoder deployment/artifacts/encoder_512x512_fp32/encoder.onnx \
        --matcher deployment/artifacts/matcher_512x512_fp32/matcher.onnx
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_image, load_roma_omega, no_autocast, set_matcher_amp  # noqa: E402
from trt_common import TrtRunner  # noqa: E402
from validate_matcher import compare  # noqa: E402


def rel_l2(test: np.ndarray, ref: np.ndarray) -> float:
    test, ref = test.astype(np.float64), ref.astype(np.float64)
    return float(np.linalg.norm(test - ref) / np.linalg.norm(ref))


def time_ms(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 2),
        "mean_ms": round(statistics.mean(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "min_ms": round(samples[0], 2),
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--encoder", type=Path, required=True, help="encoder.onnx (engine sits next to it)")
    p.add_argument("--matcher", type=Path, required=True, help="matcher.onnx (engine sits next to it)")
    p.add_argument("--img-a", default=".WxBS/v1.1/WGALBS/bridge/01.png")
    p.add_argument("--img-b", default=".WxBS/v1.1/WGALBS/bridge/02.png")
    p.add_argument("--tag", default="", help="engine name suffix used at build time (build_trt.py --tag)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    args = p.parse_args()

    prov = json.loads(args.matcher.with_name("matcher.provenance.json").read_text())
    H, W = prov["image_size"]
    feat_names = list(prov["inputs"].keys())
    out_names = list(prov["outputs"].keys())
    a, b = load_image(args.img_a, H, W), load_image(args.img_b, H, W)
    ta, tb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()

    suffix = f".{args.tag}.engine" if args.tag else ".engine"
    enc_engine = args.encoder.with_suffix(suffix)
    mat_engine = args.matcher.with_suffix(suffix)
    enc_trt = TrtRunner(enc_engine)
    mat_trt = TrtRunner(mat_engine)

    def trt_chain():
        feats = enc_trt({"img_A": ta, "img_B": tb})
        return mat_trt(feats)

    # --- accuracy -----------------------------------------------------------
    feats_trt = {n: t.float().cpu().numpy() for n, t in enc_trt({"img_A": ta, "img_B": tb}).items()}
    out_trt = {n: t.float().cpu().numpy() for n, t in trt_chain().items()}

    enc_ort = ort.InferenceSession(str(args.encoder), providers=["CPUExecutionProvider"])
    mat_ort = ort.InferenceSession(str(args.matcher), providers=["CPUExecutionProvider"])
    feats_ort = dict(zip(feat_names, enc_ort.run(None, {"img_A": a, "img_B": b})))
    out_ort = dict(zip(out_names, mat_ort.run(None, feats_ort)))
    del enc_ort, mat_ort

    model = load_roma_omega()
    matcher = model.matcher

    def torch_forward():
        return matcher(ta, tb, bidirectional=False)

    with torch.inference_mode():
        ref_bf16 = {n: torch_forward()[n].float().cpu().numpy() for n in out_names}
        set_matcher_amp(matcher, False)
        with no_autocast():
            ref_fp32 = {n: torch_forward()[n].float().cpu().numpy() for n in out_names}
        set_matcher_amp(matcher, True)

    accuracy = {
        "encoder_trt_vs_onnxruntime_rel_l2": {n: rel_l2(feats_trt[n], feats_ort[n]) for n in feat_names},
        "chain_trt_vs_onnxruntime": compare(out_trt, out_ort, H, W, "AB"),
        "chain_trt_vs_torch_fp32": compare(out_trt, ref_fp32, H, W, "AB"),
        "chain_trt_vs_torch_bf16": compare(out_trt, ref_bf16, H, W, "AB"),
        "torch_bf16_vs_torch_fp32": compare(ref_bf16, ref_fp32, H, W, "AB"),
    }

    # --- latency ------------------------------------------------------------
    feats_gpu = {n: t.clone() for n, t in enc_trt({"img_A": ta, "img_B": tb}).items()}
    latency = {
        "trt_encoder": time_ms(lambda: enc_trt({"img_A": ta, "img_B": tb}), args.warmup, args.runs),
        "trt_matcher": time_ms(lambda: mat_trt(feats_gpu), args.warmup, args.runs),
        "trt_chain": time_ms(trt_chain, args.warmup, args.runs),
    }
    with torch.inference_mode():
        latency["torch_bf16_as_shipped"] = time_ms(torch_forward, args.warmup, args.runs)
        set_matcher_amp(matcher, False)
        with no_autocast():
            latency["torch_fp32"] = time_ms(torch_forward, args.warmup, args.runs)
        set_matcher_amp(matcher, True)

    report = {
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "image_size": [H, W],
        "scope": "coarse matcher only (encoder + matcher), batch 1, one direction, no refiners",
        "engines": {
            "encoder": str(enc_engine),
            "matcher": str(mat_engine),
        },
        "accuracy": accuracy,
        "latency": latency,
    }
    out_path = args.matcher.with_name(f"trt.{args.tag}.validation.json" if args.tag else "trt.validation.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\nAccuracy")
    for n, v in accuracy["encoder_trt_vs_onnxruntime_rel_l2"].items():
        print(f"  encoder {n}: TRT vs ONNX Runtime rel_l2={v:.3e}")
    for key in ("chain_trt_vs_onnxruntime", "chain_trt_vs_torch_fp32", "chain_trt_vs_torch_bf16", "torch_bf16_vs_torch_fp32"):
        m = accuracy[key]
        print(
            f"  {key:28s} warp px mean={m['warp_px_mean']:.4f} max={m['warp_px_max']:.3f} "
            f"confident mean={m.get('warp_px_mean_confident', float('nan')):.4f} "
            f"overlap max diff={m['overlap_abs_max']:.4f}"
        )
    print("\nLatency (ms)")
    for key, m in latency.items():
        print(f"  {key:24s} median={m['median_ms']:8.2f} p95={m['p95_ms']:8.2f} min={m['min_ms']:8.2f}")
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
