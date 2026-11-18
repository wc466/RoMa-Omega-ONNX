"""Export one RoMa-Omega refinement pass (images + previous warp -> refined warp) to ONNX.

Static shape, batch 1, dynamo (torch.export) exporter. One graph holds the
VGG19-BN fine features and the patch-4, patch-2 and patch-1 ConvRefiners for a
fixed image resolution. A provenance sidecar is written alongside.

Two kinds of pass exist in upstream's pipeline:

  first pass   at the coarse resolution, fed by the coarse matcher
               (previous grid = coarse size / 4, one confidence channel)
  second pass  at the refinement resolution, fed by the first pass
               (--second-pass --prev-height/--prev-width = coarse size,
               four confidence channels; incoming precision is zeroed)

Usually run through export_pipeline.py, which passes consistent arguments.
Run from the repository root:

    python deployment/export_refiner.py --height 800 --width 800
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_rewrites import cast_constants_to_input_dtype  # noqa: E402
from common import (  # noqa: E402
    COMPUTE_DTYPE,
    RefinerStageWrapper,
    describe_files,
    local_corr_impl,
    load_roma_omega,
    refiner_io_names,
    roma_omega_checkpoint,
    sha256_file,
    tracing_context,
    write_provenance,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--height", type=int, default=800, help="image height of this pass")
    p.add_argument("--width", type=int, default=800, help="image width of this pass")
    p.add_argument("--second-pass", action="store_true", help="pass that follows a first refinement pass")
    p.add_argument("--prev-height", type=int, default=None, help="height of the incoming warp grid")
    p.add_argument("--prev-width", type=int, default=None, help="width of the incoming warp grid")
    p.add_argument("--direction", choices=["both", "ab"], default="both")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument(
        "--local-corr", choices=["upstream", "per_offset"], default="upstream",
        help="local-correlation formulation traced into the graph: upstream's single "
        "grid_sample over all window offsets, or one grid_sample per offset (same "
        "arithmetic up to rounding, see common.per_offset_local_corr)",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    bidirectional = args.direction == "both"
    if args.second_pass:
        assert args.prev_height and args.prev_width, "--second-pass needs --prev-height/--prev-width"
        prev_h, prev_w, prev_c = args.prev_height, args.prev_width, 4
    else:
        # The coarse matcher predicts on a grid of a quarter of the image size.
        prev_h, prev_w, prev_c = args.prev_height or args.height // 4, args.prev_width or args.width // 4, 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "refiner.onnx"

    print("Loading RoMa-Omega ...")
    model = load_roma_omega()
    wrapper = RefinerStageWrapper(model, bidirectional=bidirectional, second_pass=args.second_pass).eval()
    # Upstream builds grid constants on its global device, so trace there.
    dev = next(model.refiners.parameters()).device

    names_in, names_out = refiner_io_names(bidirectional)
    example = [
        torch.rand(1, 3, args.height, args.width, device=dev),
        torch.rand(1, 3, args.height, args.width, device=dev),
    ]
    for _ in ("AB", "BA")[: 2 if bidirectional else 1]:
        example.append(torch.rand(1, prev_h, prev_w, 2, device=dev) * 2 - 1)
        example.append(torch.randn(1, prev_h, prev_w, prev_c, device=dev))

    print(
        f"Exporting (dynamo, opset {args.opset}, static {args.height}x{args.width}, "
        f"previous grid {prev_h}x{prev_w}x{prev_c}) ..."
    )
    t0 = time.time()
    with tracing_context(args.precision), local_corr_impl(args.local_corr):
        program = torch.onnx.export(
            wrapper,
            tuple(example),
            dynamo=True,
            opset_version=args.opset,
            input_names=names_in,
            output_names=names_out,
        )
    program.save(str(onnx_path), external_data=True)
    print(f"Exported in {time.time() - t0:.0f}s -> {onnx_path}")

    cast = 0
    if args.precision != "fp32":
        cast = cast_constants_to_input_dtype(onnx_path)
        print(f"Cast {cast} fp32 constant(s) feeding 16-bit Conv/BatchNormalization nodes")

    graph = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(str(onnx_path))
    opsets = {o.domain or "ai.onnx": o.version for o in graph.opset_import}
    op_types = sorted({n.op_type for n in graph.graph.node})
    tensor_dtypes = dict(
        Counter(
            onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type)
            for v in graph.graph.value_info
        )
    )

    def shape_of(value) -> list:
        return [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]

    ckpt_path, ckpt_url = roma_omega_checkpoint()
    files = sorted(f for f in out_dir.iterdir() if f.name.startswith("refiner.onnx"))
    write_provenance(
        out_dir / "refiner.provenance.json",
        {
            "graph": "RoMa-Omega refinement pass (VGG19-BN fine features + ConvRefiners 4, 2, 1)",
            "exporter": {"name": "torch.onnx.export", "dynamo": True},
            "opset": opsets,
            "precision": args.precision,
            "weights_dtype": "fp32",
            "compute_dtype": COMPUTE_DTYPE[args.precision],
            "tensor_dtypes": tensor_dtypes,
            "static_shape": True,
            "bidirectional": bidirectional,
            "second_pass": args.second_pass,
            "local_correlation": args.local_corr,
            "image_size": [args.height, args.width],
            "inputs": {i.name: shape_of(i) for i in graph.graph.input},
            "outputs": {o.name: shape_of(o) for o in graph.graph.output},
            "output_convention": "warp: normalised (x, y) in [-1, 1] in the other view; "
            "confidence: [overlap logit, precision p00, p10, p11]",
            "node_count": len(graph.graph.node),
            "op_types": op_types,
            "workarounds": (
                [
                    f"{cast} fp32 constant(s) feeding 16-bit Conv/BatchNormalization nodes cast to "
                    "the node's input type (onnx_rewrites.py): the exporter leaves them fp32, "
                    "which TensorRT rejects"
                ]
                if args.precision != "fp32"
                else [
                    "torch.autocast disabled during tracing (common.no_autocast): upstream "
                    "VGG features and refiner blocks hard-code or default to bfloat16 autocast"
                ]
            ),
            "checkpoints": {
                "roma_omega": {
                    "file": str(ckpt_path),
                    "sha256": sha256_file(ckpt_path),
                    "bytes": ckpt_path.stat().st_size,
                    "source": ckpt_url,
                },
                "vgg19_bn": "torchvision VGG19_BN_Weights.IMAGENET1K_V1 as loaded by upstream; "
                "the fine-feature weights saved in the RoMa-Omega checkpoint override it",
            },
            "files": describe_files(files),
        },
    )
    print(f"Provenance -> {out_dir / 'refiner.provenance.json'}")
    print(f"{len(graph.graph.node)} nodes, op types: {', '.join(op_types)}")


if __name__ == "__main__":
    main()
