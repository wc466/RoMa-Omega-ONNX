"""Build the TensorRT engines of a RoMa-Omega pipeline.

Runs build_trt.py on every graph of one pipeline configuration and precision
(see pipeline.py for the configuration flags; defaults are upstream's
``precise`` setting). Engines that already exist are skipped unless --force.

Run from the repository root:

    python deployment/build_pipeline_trt.py --precision bf16
    python deployment/build_pipeline_trt.py --precision fp32 --no-tf32

With --no-tf32 the engines are named ``<graph>.notf32.engine`` (variant
``trt_fp32``); without it ``<graph>.engine`` (``trt_fp32_tf32``, ``trt_bf16``,
``trt_fp16``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import PRECISIONS, PipelineConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    PipelineConfig.add_arguments(p)
    p.add_argument("--precision", choices=PRECISIONS, default="bf16")
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 (only matters for fp32 graphs)")
    p.add_argument("--stages", default=None, help="comma-separated subset of stages to build")
    p.add_argument("--artifacts", type=Path, default=Path("deployment/artifacts"))
    p.add_argument("--force", action="store_true", help="rebuild engines that already exist")
    args = p.parse_args()

    cfg = PipelineConfig.from_args(args)
    stages = [s for s in (args.stages.split(",") if args.stages else cfg.stages) if s]
    suffix = ".notf32.engine" if args.no_tf32 else ".engine"
    failed = []
    for stage in stages:
        onnx_path = cfg.graph_path(stage, args.precision, args.artifacts)
        assert onnx_path.exists(), f"missing {onnx_path}; run export_pipeline.py first"
        engine = onnx_path.with_name(onnx_path.stem + suffix)
        if engine.exists() and not args.force:
            print(f"[{stage}] exists, skipping: {engine}")
            continue
        cmd = [sys.executable, str(HERE / "build_trt.py"), "--onnx", str(onnx_path)]
        if args.no_tf32:
            cmd += ["--no-tf32", "--tag", "notf32"]
        print(f"[{stage}] building {engine}", flush=True)
        if subprocess.run(cmd).returncode != 0 or not engine.exists():
            failed.append(stage)
            print(f"[{stage}] BUILD FAILED (see {engine}.buildlog.json)")
    if failed:
        raise SystemExit(f"failed stages: {', '.join(failed)}")


if __name__ == "__main__":
    main()
