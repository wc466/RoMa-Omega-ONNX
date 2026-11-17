"""Latency and memory at batch 1 for every coarse-matcher backend.

Each variant runs in a fresh subprocess, on an otherwise idle GPU, with one
real image pair at 512x512. Scope is the network only (encoder + matcher):
no image loading, resizing or match sampling. ``torch_bf16_full`` additionally
times the released full pipeline forward (800 coarse / 1280 refine,
bidirectional, refiners) for context.

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
        --variants torchtrt_fp32_coarse,torchtrt_bf16_coarse --append
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

DEFAULT_VARIANTS = (
    "torch_bf16_full",
    "torch_fp32_coarse",
    "torch_bf16_coarse",
    "onnx_fp32_coarse",
    "trt_fp32_coarse",
    "trt_fp32_tf32_coarse",
    "trt_bf16_coarse",
    "trt_fp16enc_bf16mat_coarse",
    "trt_fp16_coarse",
)
ARTIFACTS = Path("deployment/artifacts")
SIZE = "512x512"


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


def child(variant: str, warmup: int, runs: int) -> dict:
    import torch

    from common import load_image
    from matchers import GRAPH_VARIANTS, TORCHTRT_VARIANTS

    torch.cuda.init()
    torch.zeros(1, device="cuda")
    free0 = torch.cuda.mem_get_info()[0]

    a = torch.from_numpy(load_image(".WxBS/v1.1/WGALBS/bridge/01.png", 512, 512)).cuda()
    b = torch.from_numpy(load_image(".WxBS/v1.1/WGALBS/bridge/02.png", 512, 512)).cuda()
    disk, parts, extra = 0.0, {}, {}

    def timed(fn, n_warm=warmup, n=runs):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
        return stats(out)

    if variant in GRAPH_VARIANTS:
        enc_prec, mat_prec, suffix = GRAPH_VARIANTS[variant]
        enc_path = ARTIFACTS / f"encoder_{SIZE}_{enc_prec}" / f"encoder{suffix}"
        mat_path = ARTIFACTS / f"matcher_{SIZE}_{mat_prec}" / f"matcher{suffix}"
        if suffix == ".onnx":
            import onnxruntime as ort

            disk = sum(Path(str(p) + ".data").stat().st_size + p.stat().st_size for p in (enc_path, mat_path))
            enc = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
            mat = ort.InferenceSession(str(mat_path), providers=["CPUExecutionProvider"])
            an, bn = a.cpu().numpy(), b.cpu().numpy()
            names = [i.name for i in mat.get_inputs()]
            feats = enc.run(None, {"img_A": an, "img_B": bn})
            parts["encoder"] = timed(lambda: enc.run(None, {"img_A": an, "img_B": bn}), 1, 5)
            parts["matcher"] = timed(lambda: mat.run(None, dict(zip(names, feats))), 1, 5)
            total = timed(
                lambda: mat.run(None, dict(zip(names, enc.run(None, {"img_A": an, "img_B": bn})))), 1, 5
            )
            extra["device"] = "CPU (20 cores)"
        else:
            from trt_common import TrtRunner

            disk = enc_path.stat().st_size + mat_path.stat().st_size
            enc, mat = TrtRunner(enc_path), TrtRunner(mat_path)
            feats = {k: v.clone() for k, v in enc({"img_A": a, "img_B": b}).items()}
            parts["encoder"] = timed(lambda: enc({"img_A": a, "img_B": b}))
            parts["matcher"] = timed(lambda: mat(feats))
            total = timed(lambda: mat(enc({"img_A": a, "img_B": b})))
            out = mat(enc({"img_A": a, "img_B": b}))
            extra["engine_device_memory_mb"] = round(
                (enc.engine.device_memory_size_v2 + mat.engine.device_memory_size_v2) / 2**20, 1
            )
            extra["output_finite"] = bool(torch.isfinite(out["warp_AB"].float()).all())
    else:
        from common import (
            EncoderWrapper,
            force_autocast_dtype,
            load_roma_omega,
            no_autocast,
            roma_omega_checkpoint,
            set_matcher_amp,
        )

        model = load_roma_omega()
        matcher = model.matcher
        disk = Path("vggtomega.pt").stat().st_size + roma_omega_checkpoint()[0].stat().st_size
        torch.cuda.reset_peak_memory_stats()
        if variant in TORCHTRT_VARIANTS:
            from matchers import TorchTrtCoarseMatcher

            gm = TorchTrtCoarseMatcher(matcher, TORCHTRT_VARIANTS[variant], 512, 512)
            extra["matcher_backend"] = gm.matcher_backend
            with torch.inference_mode(), gm._ctx():
                feats = [f.clone() for f in gm.encoder(a, b)]
                parts["encoder"] = timed(lambda: gm.encoder(a, b))
                parts["matcher"] = timed(lambda: gm.matcher(*feats))
                total = timed(lambda: gm.matcher(*gm.encoder(a, b)))
        elif variant == "torch_bf16_full":
            import torch.nn.functional as F

            model.stage = "inference"
            model.apply_setting("precise")

            def resize(x, h, w):
                return F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)

            lr = [resize(x, model.H_lr, model.W_lr) for x in (a, b)]
            hr = [resize(x, model.H_hr, model.W_hr) for x in (a, b)]
            with torch.inference_mode():
                total = timed(lambda: model(lr[0], lr[1], hr[0], hr[1]), 3, 15)
            extra["setting"] = f"{model.H_lr} coarse / {model.H_hr} refine, bidirectional, refiners"
        else:
            import contextlib

            amp = variant != "torch_fp32_coarse"
            ctx = {
                "torch_fp32_coarse": no_autocast,
                "torch_fp16_coarse": lambda: force_autocast_dtype(torch.float16),
            }.get(variant, contextlib.nullcontext)
            set_matcher_amp(matcher, amp)
            encoder = EncoderWrapper(matcher.vggt.aggregator, amp=amp).eval()
            with torch.inference_mode(), ctx():
                parts["encoder"] = timed(lambda: encoder(a, b))
                total = timed(lambda: matcher(a, b, bidirectional=False))
            # No separate matcher timing in PyTorch: Matcher.forward includes the encoder.
            parts["matcher"] = {
                "median_ms": round(total["median_ms"] - parts["encoder"]["median_ms"], 2),
                "derived": "total - encoder",
            }
        extra["torch_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

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


def run_variant(variant: str, warmup: int, runs: int) -> dict:
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
        [sys.executable, __file__, "--child", variant, "--warmup", str(warmup), "--runs", str(runs)],
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
    p.add_argument("--out", type=Path, default=Path("deployment/results/runtime_512x512.json"))
    p.add_argument("--append", action="store_true", help="merge into an existing results file")
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child:
        print("RESULT " + json.dumps(child(args.child, args.warmup, args.runs)), flush=True)
        return

    import platform

    import torch

    records = {}
    if args.append and args.out.exists():
        records = {r["variant"]: r for r in json.loads(args.out.read_text())["variants"]}
    for variant in [v for v in args.variants.split(",") if v]:
        rec = run_variant(variant, args.warmup, args.runs)
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
                "image_size": [512, 512],
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
