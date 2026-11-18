"""Compare pipeline variants against upstream PyTorch on one image pair.

For a pipeline configuration (pipeline.py; defaults are upstream's ``precise``
setting) every requested variant is run through upstream's own ``match`` and
its final warp is compared with a reference variant (default: PyTorch fp32,
every autocast disabled), in pixels of the output grid, over the pixels the
reference considers overlapping (overlap > 0.5).

Run from the repository root:

    python deployment/validate_pipeline.py --variants torch_bf16,onnx_gpu_fp32,trt_fp32,trt_bf16
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import force_autocast_dtype, no_autocast  # noqa: E402
from matchers import TORCH_VARIANTS, VARIANTS, RoMaOmegaVariants  # noqa: E402
from pipeline import PipelineConfig  # noqa: E402


def predict(models: RoMaOmegaVariants, variant: str, img_a: str, img_b: str) -> tuple[dict, float]:
    models.select(variant)
    precision = TORCH_VARIANTS.get(variant)
    ctx = {"fp32": no_autocast, "fp16": lambda: force_autocast_dtype(torch.float16)}.get(
        precision, contextlib.nullcontext
    )
    with torch.inference_mode(), ctx():
        models.model.match(img_a, img_b)  # warm-up (lazy initialisation, cache)
        if variant not in TORCH_VARIANTS:
            models.pipeline(variant).use_cache = False
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = models.model.match(img_a, img_b)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
    return {k: v.float().cpu() for k, v in preds.items() if v is not None}, ms


def compare(ref: dict, out: dict, directions) -> dict:
    stats = {}
    for d in directions:
        H, W = ref[f"warp_{d}"].shape[1:3]
        scale = torch.tensor([W / 2, H / 2])
        err = ((out[f"warp_{d}"] - ref[f"warp_{d}"]) * scale).norm(dim=-1)[0]
        mask = ref[f"overlap_{d}"][0, ..., 0] > 0.5
        finite = bool(torch.isfinite(out[f"warp_{d}"]).all())
        e = err[mask]
        stats[d] = {
            "finite": finite,
            "overlap_pixels": int(mask.sum()),
            "warp_err_px_mean": float(e.mean()) if finite and len(e) else None,
            "warp_err_px_median": float(e.median()) if finite and len(e) else None,
            "warp_err_px_p99": float(e.quantile(0.99)) if finite and len(e) and len(e) < 16_000_000 else None,
            "warp_err_px_max": float(e.max()) if finite and len(e) else None,
            "overlap_abs_diff_mean": float((out[f"overlap_{d}"] - ref[f"overlap_{d}"]).abs().mean()),
        }
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    PipelineConfig.add_arguments(p)
    p.add_argument("--variants", default="torch_bf16,onnx_gpu_fp32,trt_fp32,trt_fp32_tf32,trt_bf16")
    p.add_argument("--reference", default="torch_fp32")
    p.add_argument("--img-a", default=".WxBS/v1.1/WGALBS/bridge/01.png")
    p.add_argument("--img-b", default=".WxBS/v1.1/WGALBS/bridge/02.png")
    p.add_argument("--artifacts", type=Path, default=Path("deployment/artifacts"))
    p.add_argument("--out-dir", type=Path, default=Path("deployment/results/validation"))
    args = p.parse_args()

    cfg = PipelineConfig.from_args(args)
    variants = [v for v in args.variants.split(",") if v]
    assert all(v in VARIANTS for v in variants), variants
    models = RoMaOmegaVariants(cfg, args.artifacts)
    ref, ref_ms = predict(models, args.reference, args.img_a, args.img_b)
    report = {
        "config": cfg.describe(),
        "pair": [args.img_a, args.img_b],
        "reference": args.reference,
        "output_grid": list(ref["warp_AB"].shape[1:3]),
        "variants": {},
    }
    print(f"config {cfg.tag}; reference {args.reference} ({ref_ms:.0f} ms); "
          f"output grid {report['output_grid']}")
    for variant in variants:
        try:
            out, ms = predict(models, variant, args.img_a, args.img_b)
        except Exception as e:  # noqa: BLE001 - reported per variant
            report["variants"][variant] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(f"{variant:18s} ERROR {type(e).__name__}: {str(e)[:200]}")
            models.release(variant)
            continue
        stats = compare(ref, out, cfg.directions)
        report["variants"][variant] = {"match_ms": round(ms, 1), **models.describe(variant), "vs_reference": stats}
        for d, s in stats.items():
            if s["warp_err_px_mean"] is None:
                print(f"{variant:18s} {d}  output not finite")
                continue
            print(
                f"{variant:18s} {d}  warp err px: mean {s['warp_err_px_mean']:.4f} "
                f"median {s['warp_err_px_median']:.4f} max {s['warp_err_px_max']:.3f} | "
                f"match {ms:.0f} ms"
            )
        models.release(variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"validation__{cfg.tag}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
