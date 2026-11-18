"""Export the RoMa-Omega coarse matcher (encoder features -> warp, confidence) to ONNX.

Static shape, batch 1, dynamo (torch.export) exporter, no refiners. Consumes
the outputs of the encoder graph from export_encoder.py at the same input
resolution. A provenance sidecar is written alongside.

--precision fp32 (default) gives a pure fp32 graph. --precision bf16 preserves
the released model's precision: fp32 weights, bf16 computation exactly where
upstream's autocast applies (matcher body and DPT head, final conv in fp32),
with fp32 inputs and outputs.

Run from the repository root:

    python deployment/export_matcher.py --height 800 --width 800
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
from onnx_rewrites import rewrite_deconv_as_depth_to_space  # noqa: E402
from common import (  # noqa: E402
    COMPUTE_DTYPE,
    tracing_context,
    EMBED_DIM,
    PATCH_SIZE,
    MatcherWrapper,
    describe_files,
    load_roma_omega,
    matcher_output_names,
    output_names,
    roma_omega_checkpoint,
    sha256_file,
    write_provenance,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--height", type=int, default=800, help="image height fed to the encoder")
    p.add_argument("--width", type=int, default=800, help="image width fed to the encoder")
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
        "--direction", choices=["both", "ab"], default="both",
        help="both: also output warp_BA/confidence_BA; ab: A->B only",
    )
    p.add_argument(
        "--rewrite-deconv",
        choices=["auto", "on", "off"],
        default="auto",
        help="rewrite ConvTranspose as Conv + DepthToSpace (exact). auto: only for "
        "bf16, where TensorRT 11.4 cannot build ConvTranspose",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    args.bidirectional = args.direction == "both"
    if args.height % PATCH_SIZE or args.width % PATCH_SIZE:
        raise SystemExit(f"height and width must be multiples of {PATCH_SIZE}")
    H_p, W_p = args.height // PATCH_SIZE, args.width // PATCH_SIZE

    suffix = "_bidir" if args.bidirectional else ""
    out_dir = args.out_dir or Path(
        f"deployment/artifacts/matcher_{args.height}x{args.width}_{args.precision}{suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "matcher.onnx"

    print("Loading RoMa-Omega ...")
    model = load_roma_omega()
    amp = args.precision != "fp32"
    wrapper = MatcherWrapper(model.matcher, bidirectional=args.bidirectional, amp=amp).eval()
    # Upstream builds grid constants on its global device, so trace there.
    dev = next(model.matcher.parameters()).device

    names_in = output_names(wrapper.layers)
    names_out = matcher_output_names(args.bidirectional)
    example = tuple(
        torch.randn(1, 2, H_p, W_p, 2 * EMBED_DIM, device=dev) for _ in names_in
    )

    print(f"Exporting (dynamo, opset {args.opset}, static {H_p}x{W_p} tokens) ...")
    t0 = time.time()
    # fp32: silence every autocast (the DPT head hard-codes one).
    # bf16: leave upstream's autocast in place so the graph mirrors it.
    # fp16: same placement, dtype forced to float16.
    with tracing_context(args.precision):
        program = torch.onnx.export(
            wrapper,
            example,
            dynamo=True,
            opset_version=args.opset,
            input_names=names_in,
            output_names=names_out,
        )
    program.save(str(onnx_path), external_data=True)
    print(f"Exported in {time.time() - t0:.0f}s -> {onnx_path}")

    rewritten = 0
    if args.rewrite_deconv == "on" or (args.rewrite_deconv == "auto" and amp):
        rewritten = rewrite_deconv_as_depth_to_space(onnx_path)
        print(f"Rewrote {rewritten} ConvTranspose node(s) as Conv + DepthToSpace")

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
    files = sorted(f for f in out_dir.iterdir() if f.name.startswith("matcher.onnx"))
    write_provenance(
        out_dir / "matcher.provenance.json",
        {
            "graph": "RoMa-Omega coarse matcher (cross-view transformer + DPT head), no refiners",
            "exporter": {"name": "torch.onnx.export", "dynamo": True},
            "opset": opsets,
            "precision": args.precision,
            "weights_dtype": "fp32",
            "compute_dtype": COMPUTE_DTYPE[args.precision],
            "tensor_dtypes": tensor_dtypes,
            "static_shape": True,
            "bidirectional": args.bidirectional,
            "image_size": [args.height, args.width],
            "inputs": {i.name: shape_of(i) for i in graph.graph.input},
            "outputs": {o.name: shape_of(o) for o in graph.graph.output},
            "output_convention": "warp: normalised (x, y) in [-1, 1] in the other view; "
            "confidence: overlap logit (apply sigmoid)",
            "cached_layers": list(wrapper.layers),
            "node_count": len(graph.graph.node),
            "op_types": op_types,
            "workarounds": (
                []
                if amp
                else [
                    "torch.autocast disabled during tracing (common.no_autocast): "
                    "upstream DPT head hard-codes bfloat16 autocast"
                ]
            )
            + (
                [
                    f"{rewritten} ConvTranspose node(s) rewritten as Conv + DepthToSpace "
                    "(onnx_rewrites.py): TensorRT 11.4 has no bf16 ConvTranspose"
                ]
                if rewritten
                else []
            ),
            "checkpoints": {
                "roma_omega": {
                    "file": str(ckpt_path),
                    "sha256": sha256_file(ckpt_path),
                    "bytes": ckpt_path.stat().st_size,
                    "source": ckpt_url,
                }
            },
            "files": describe_files(files),
        },
    )
    print(f"Provenance -> {out_dir / 'matcher.provenance.json'}")
    print(f"{len(graph.graph.node)} nodes, op types: {', '.join(op_types)}")


if __name__ == "__main__":
    main()
