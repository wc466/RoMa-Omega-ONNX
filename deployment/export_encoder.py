"""Export the frozen VGGT-Omega pair encoder of RoMa-Omega to ONNX.

Static shape, batch 1, dynamo (torch.export) exporter. Weights are stored as
fp32 external data next to the graph (the encoder is ~1B parameters, well over
the 2 GB protobuf limit). A provenance sidecar is written alongside.

--precision fp32 (default) gives a pure fp32 graph. --precision bf16 preserves
the released model's precision: fp32 weights, computed under bf16 autocast as
upstream does, with fp32 inputs and outputs.

Run from the repository root:

    python deployment/export_encoder.py --height 512 --width 512
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
from common import (  # noqa: E402
    COMPUTE_DTYPE,
    tracing_context,
    DEFAULT_LAYERS,
    PATCH_SIZE,
    VGGT_OMEGA_LICENSE,
    VGGT_OMEGA_SOURCE,
    describe_files,
    load_encoder,
    output_names,
    parse_layers,
    sha256_file,
    write_provenance,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default="vggtomega.pt", help="VGGT-Omega checkpoint")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="fp32: pure fp32 graph. bf16: the released model's precision - fp32 "
        "weights, bf16 computation where upstream autocast applies, fp32 I/O. "
        "fp16: same placement with IEEE float16",
    )
    p.add_argument(
        "--layers",
        type=parse_layers,
        default=DEFAULT_LAYERS,
        help="cached aggregator layers to output, shallow->deep indices into "
        "(4, 11, 17, 23); default is what the released matcher consumes",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if args.height % PATCH_SIZE or args.width % PATCH_SIZE:
        raise SystemExit(f"height and width must be multiples of {PATCH_SIZE}")

    out_dir = args.out_dir or Path(
        f"deployment/artifacts/encoder_{args.height}x{args.width}_{args.precision}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "encoder.onnx"

    print(f"Loading {args.checkpoint} ...")
    amp = args.precision != "fp32"
    encoder = load_encoder(args.checkpoint, args.layers, amp=amp)

    # fp32: export on CPU, no autocast, no device-specific constants.
    # bf16: trace on CUDA, where upstream's bf16 autocast actually applies.
    dev = "cuda" if amp else "cpu"
    encoder = encoder.to(dev)
    example = (
        torch.rand(1, 3, args.height, args.width, device=dev),
        torch.rand(1, 3, args.height, args.width, device=dev),
    )
    names_out = output_names(args.layers)

    print(f"Exporting (dynamo, opset {args.opset}, static {args.height}x{args.width}) ...")
    t0 = time.time()
    with tracing_context(args.precision):
        program = torch.onnx.export(
            encoder,
            example,
            dynamo=True,
            opset_version=args.opset,
            input_names=["img_A", "img_B"],
            output_names=names_out,
        )
    program.save(str(onnx_path), external_data=True)
    print(f"Exported in {time.time() - t0:.0f}s -> {onnx_path}")

    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(str(onnx_path))
    opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    op_types = sorted({n.op_type for n in model.graph.node})
    tensor_dtypes = dict(
        Counter(
            onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type)
            for v in model.graph.value_info
        )
    )

    def shape_of(value) -> list:
        return [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]

    files = sorted(f for f in out_dir.iterdir() if f.name.startswith("encoder.onnx"))
    write_provenance(
        out_dir / "encoder.provenance.json",
        {
            "graph": "RoMa-Omega pair encoder (VGGT-Omega aggregator)",
            "exporter": {"name": "torch.onnx.export", "dynamo": True},
            "opset": opsets,
            "precision": args.precision,
            "weights_dtype": "fp32",
            "compute_dtype": COMPUTE_DTYPE[args.precision],
            "tensor_dtypes": tensor_dtypes,
            "static_shape": True,
            "inputs": {i.name: shape_of(i) for i in model.graph.input},
            "outputs": {o.name: shape_of(o) for o in model.graph.output},
            "input_convention": "RGB in [0, 1]; ImageNet normalisation is inside the graph",
            "cached_layers": list(args.layers),
            "node_count": len(model.graph.node),
            "op_types": op_types,
            "checkpoints": {
                "vggt_omega": {
                    "file": str(args.checkpoint),
                    "sha256": sha256_file(args.checkpoint),
                    "bytes": Path(args.checkpoint).stat().st_size,
                    "source": VGGT_OMEGA_SOURCE,
                    "license": VGGT_OMEGA_LICENSE,
                }
            },
            "files": describe_files(files),
        },
    )
    print(f"Provenance -> {out_dir / 'encoder.provenance.json'}")
    print(f"{len(model.graph.node)} nodes, op types: {', '.join(op_types)}")


if __name__ == "__main__":
    main()
