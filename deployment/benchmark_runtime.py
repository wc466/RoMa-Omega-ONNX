"""Latency and memory at batch 1 for every backend of one pipeline configuration.

Each variant runs in a fresh subprocess, on an otherwise idle GPU, with one
real image pair. Scope is the network only (every graph of the pipeline, see
pipeline.py; defaults are upstream's ``precise`` setting): no image loading,
resizing or match sampling.

Memory, per variant:
  system_peak_mb  peak drop in system MemAvailable while the process runs.
                  The DGX Spark has unified memory, so this covers host and
                  GPU allocations together (weights, engine, activations,
                  framework overhead).
  gpu_delta_mb    drop in free GPU memory between CUDA initialisation and the
                  end of inference (excludes the CUDA context itself).
  torch_peak_mb   torch.cuda.max_memory_allocated (PyTorch tensors only).
  disk_mb         size of the weights the variant loads from disk.

Run from the repository root:

    python deployment/benchmark_runtime.py
    .venv-torchtrt/bin/python deployment/benchmark_runtime.py \
        --variants torchtrt_fp32,torchtrt_bf16 --append
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import PipelineConfig  # noqa: E402

DEFAULT_VARIANTS = (
    "torch_bf16",
    "torch_fp32",
    "onnx_gpu_fp32",
    "trt_fp32",
    "trt_fp32_tf32",
    "trt_bf16",
    "trt_fp16enc_bf16",
    "trt_fp16",
)
IMAGES = (".WxBS/v1.1/WGALBS/bridge/01.png", ".WxBS/v1.1/WGALBS/bridge/02.png")
# Variants slower than this per pair are timed with fewer runs.
SLOW_MS, SLOW_RUNS = 3000.0, 5


def mem_available_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable not found")


def stats(samples: list[float]) -> dict:
    samples = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 2),
        "mean_ms": round(statistics.mean(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "min_ms": round(samples[0], 2),
        "runs": len(samples),
    }


def child(variant: str, cfg: PipelineConfig, artifacts: Path, warmup: int, runs: int) -> dict:
    import contextlib

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from matchers import (
        ONNX_VARIANTS,
        TORCH_VARIANTS,
        TRT_VARIANTS,
        build_onnx_pipeline,
        build_torchtrt_pipeline,
        build_trt_pipeline,
        trt_engine_paths,
    )

    torch.cuda.init()
    torch.zeros(1, device="cuda")
    free0 = torch.cuda.mem_get_info()[0]

    def load(path, size):
        # Same preprocessing as VGGTRoMa.match: [0, 1] RGB, antialiased bicubic resize.
        img = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).cuda()
        return F.interpolate(img.float()[None] / 255.0, size=size, mode="bicubic",
                             align_corners=False, antialias=True)

    lr = [load(p, cfg.coarse) for p in IMAGES]
    hr = [load(p, cfg.refine) for p in IMAGES] if cfg.refine is not None else [None, None]
    disk, parts, extra = 0.0, {}, {}

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        first = (time.perf_counter() - t0) * 1e3
        n_warm, n = (0, SLOW_RUNS) if first > SLOW_MS else (warmup, runs)
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
        return stats(out), n

    def file_size(p: Path) -> int:
        data = Path(str(p) + ".data")
        return p.stat().st_size + (data.stat().st_size if data.exists() else 0)

    if variant in TORCH_VARIANTS:
        from common import force_autocast_dtype, load_roma_omega, no_autocast, roma_omega_checkpoint, set_matcher_amp
        from matchers import apply_config

        model = load_roma_omega()
        disk = Path("vggtomega.pt").stat().st_size + roma_omega_checkpoint()[0].stat().st_size
        precision = TORCH_VARIANTS[variant]
        set_matcher_amp(model.matcher, precision != "fp32")
        apply_config(model, cfg)
        ctx = {"fp32": no_autocast, "fp16": lambda: force_autocast_dtype(torch.float16)}.get(
            precision, contextlib.nullcontext
        )
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), ctx():
            total, _ = timed(lambda: model(lr[0], lr[1], img_A_hr=hr[0], img_B_hr=hr[1]))
            out = model(lr[0], lr[1], img_A_hr=hr[0], img_B_hr=hr[1])
        extra["torch_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    else:
        if variant in ONNX_VARIANTS:
            pipe = build_onnx_pipeline(cfg, variant, artifacts)
            disk = sum(file_size(cfg.graph_path(s, "fp32", artifacts)) for s in cfg.stages)
        elif variant in TRT_VARIANTS:
            pipe = build_trt_pipeline(cfg, variant, artifacts)
            disk = sum(p.stat().st_size for p in trt_engine_paths(cfg, variant, artifacts).values())
            extra["engine_device_memory_mb"] = round(
                sum(r.engine.device_memory_size_v2 for r in pipe.runners.values()) / 2**20, 1
            )
        else:
            from common import load_roma_omega, roma_omega_checkpoint

            model = load_roma_omega()
            disk = Path("vggtomega.pt").stat().st_size + roma_omega_checkpoint()[0].stat().st_size
            torch.cuda.reset_peak_memory_stats()
            pipe = build_torchtrt_pipeline(cfg, variant, model)
            extra["torch_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        extra["backends"] = pipe.backends
        total, n = timed(lambda: pipe.run(lr[0], lr[1], hr[0], hr[1]))
        # Per-stage times from a separate pass (synchronising between stages).
        pipe.timing = True
        for _ in range(n):
            pipe.run(lr[0], lr[1], hr[0], hr[1])
        parts = {s: stats(v) for s, v in pipe.seconds.items()}
        out = pipe.run(lr[0], lr[1], hr[0], hr[1])
    extra["output_finite"] = bool(
        all(torch.isfinite(out[f"warp_{d}"].float()).all() for d in cfg.directions)
    )

    torch.cuda.synchronize()
    free1 = torch.cuda.mem_get_info()[0]
    return {
        "variant": variant,
        "total": total,
        "parts": parts,
        "gpu_delta_mb": round((free0 - free1) / 2**20, 1),
        "disk_mb": round(disk / 2**20, 1),
        **extra,
    }


def run_variant(variant: str, cfg: PipelineConfig, artifacts: Path, warmup: int, runs: int) -> dict:
    time.sleep(3)  # let the previous process release memory
    baseline = mem_available_mb()
    low = [baseline]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            low[0] = min(low[0], mem_available_mb())
            time.sleep(0.05)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    proc = subprocess.run(
        [sys.executable, __file__, "--child", variant, "--warmup", str(warmup), "--runs", str(runs),
         "--artifacts", str(artifacts), *cfg.to_args()],
        capture_output=True,
        text=True,
    )
    stop.set()
    t.join()
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        return {"variant": variant, "error": proc.stderr.strip().splitlines()[-1][:300] if proc.stderr else "failed"}
    rec = json.loads(lines[-1][len("RESULT "):])
    rec["system_peak_mb"] = round(baseline - low[0], 1)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    PipelineConfig.add_arguments(p)
    p.add_argument("--artifacts", type=Path, default=Path("deployment/artifacts"))
    p.add_argument("--out", type=Path, default=None, help="default: deployment/results/runtime__<config>.json")
    p.add_argument("--append", action="store_true", help="merge into an existing results file")
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    cfg = PipelineConfig.from_args(args)
    args.out = args.out or Path(f"deployment/results/runtime__{cfg.tag}.json")
    if args.child:
        print("RESULT " + json.dumps(child(args.child, cfg, args.artifacts, args.warmup, args.runs)), flush=True)
        return

    import platform

    import torch

    records = {}
    if args.append and args.out.exists():
        records = {r["variant"]: r for r in json.loads(args.out.read_text())["variants"]}
    for variant in [v for v in args.variants.split(",") if v]:
        rec = run_variant(variant, cfg, args.artifacts, args.warmup, args.runs)
        records[variant] = rec
        if "error" in rec:
            print(f"{variant:28s} ERROR {rec['error']}", flush=True)
        else:
            print(
                f"{variant:28s} total {rec['total']['median_ms']:8.1f} ms | "
                f"system peak {rec['system_peak_mb']:8.0f} MB | gpu delta {rec['gpu_delta_mb']:8.0f} MB | "
                f"disk {rec['disk_mb']:7.0f} MB",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "machine": platform.machine(),
                "config": cfg.describe(),
                "batch": 1,
                "variants": list(records.values()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
