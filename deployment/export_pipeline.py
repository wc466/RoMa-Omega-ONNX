"""Export every graph of a RoMa-Omega pipeline to ONNX with consistent settings.

Runs export_encoder.py, export_matcher.py and export_refiner.py (once per
refinement pass) for one pipeline configuration and precision. The defaults
reproduce upstream's ``precise`` setting; see pipeline.py for the flags.

Run from the repository root:

    python deployment/export_pipeline.py --precision bf16
    python deployment/export_pipeline.py --precision bf16 --coarse-size 512 \
        --refine-size none --direction ab --refiners off      # coarse only

Graphs that already exist are skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import PRECISIONS, PipelineConfig  # noqa: E402


def stage_command(
    cfg: PipelineConfig, stage: str, precision: str, artifacts: Path, opset: int, local_corr: str
) -> list[str]:
    out_dir = cfg.graph_dir(stage, precision, artifacts)
    ch, cw = cfg.coarse
    direction = "both" if cfg.bidirectional else "ab"
    common = ["--precision", precision, "--opset", str(opset), "--out-dir", str(out_dir)]
    if stage == "encoder":
        return ["export_encoder.py", "--height", str(ch), "--width", str(cw), *common]
    if stage == "matcher":
        return ["export_matcher.py", "--height", str(ch), "--width", str(cw), "--direction", direction, *common]
    common += ["--local-corr", local_corr]
    if stage == "refiner":
        return ["export_refiner.py", "--height", str(ch), "--width", str(cw), "--direction", direction, *common]
    rh, rw = cfg.refine
    return [
        "export_refiner.py", "--height", str(rh), "--width", str(rw), "--second-pass",
        "--prev-height", str(ch), "--prev-width", str(cw), "--direction", direction, *common,
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    PipelineConfig.add_arguments(p)
    p.add_argument("--precision", choices=PRECISIONS, default="bf16",
                   help="bf16 is the released model's compute precision")
    p.add_argument("--stages", default=None, help="comma-separated subset of stages to export")
    p.add_argument(
        "--local-corr", choices=["per_offset", "upstream"], default="per_offset",
        help="refiner local-correlation formulation (export_refiner.py). per_offset is "
        "arithmetically the same as upstream's up to rounding and about 4x faster in TensorRT",
    )
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--artifacts", type=Path, default=Path("deployment/artifacts"))
    p.add_argument("--force", action="store_true", help="re-export graphs that already exist")
    args = p.parse_args()

    cfg = PipelineConfig.from_args(args)
    stages = [s for s in (args.stages.split(",") if args.stages else cfg.stages) if s]
    assert all(s in cfg.stages for s in stages), (stages, cfg.stages)
    print(f"Pipeline {cfg.tag} ({args.precision}): {', '.join(stages)}")
    for stage in stages:
        onnx_path = cfg.graph_path(stage, args.precision, args.artifacts)
        if onnx_path.exists() and not args.force:
            print(f"[{stage}] exists, skipping: {onnx_path}")
            continue
        script, *rest = stage_command(cfg, stage, args.precision, args.artifacts, args.opset, args.local_corr)
        print(f"[{stage}] {script} {' '.join(rest)}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *rest], check=True)
    print("Graphs:")
    for stage in cfg.stages:
        print(f"  {stage:9s} {cfg.graph_path(stage, args.precision, args.artifacts)}")


if __name__ == "__main__":
    main()
