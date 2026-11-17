"""Build a TensorRT engine from an exported ONNX graph.

TensorRT 11 networks are always strongly typed: the engine's precision is the
precision of the ONNX graph (fp32 graph -> fp32 engine). Every parser error,
builder warning and per-op support result is recorded in a build log next to
the engine, together with a provenance sidecar.

Run from the repository root:

    python deployment/build_trt.py \
        --onnx deployment/artifacts/matcher_512x512_fp32/matcher.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnx
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256_file, write_provenance  # noqa: E402
from trt_common import CollectingLogger  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--opt-level", type=int, default=None, help="builder_optimization_level (0-5)")
    p.add_argument(
        "--no-tf32",
        action="store_true",
        help="disable TF32 (on by default in TensorRT): full fp32 matmul/conv precision",
    )
    p.add_argument("--tag", default="", help="engine name suffix, e.g. 'notf32' -> <stem>.notf32.engine")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    stem = args.onnx.stem
    name = f"{stem}.{args.tag}" if args.tag else stem
    engine_path = args.onnx.with_name(f"{name}.engine")
    log_path = args.onnx.with_name(f"{name}.engine.buildlog.json")
    onnx_prov = json.loads(args.onnx.with_name(f"{stem}.provenance.json").read_text())

    logger = CollectingLogger(verbose=args.verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    # Per-op support, independent of whether the whole graph parses.
    graph = onnx.load(str(args.onnx), load_external_data=False)
    op_types = sorted({n.op_type for n in graph.graph.node})
    op_support = {op: bool(parser.supports_operator(op)) for op in op_types}
    unsupported = [op for op, ok in op_support.items() if not ok]
    print(f"{len(op_types)} op types, unsupported by parser: {unsupported or 'none'}")

    # parse_from_file resolves the external weight file next to the graph.
    t0 = time.time()
    parsed = parser.parse_from_file(str(args.onnx))
    parser_errors = [
        {
            "node": parser.get_error(i).node_name(),
            "op": parser.get_error(i).node_operator(),
            "code": str(parser.get_error(i).code()),
            "desc": parser.get_error(i).desc(),
        }
        for i in range(parser.num_errors)
    ]
    build_log = {
        "onnx": str(args.onnx),
        "tensorrt": trt.__version__,
        "op_support": op_support,
        "unsupported_ops": unsupported,
        "parsed": bool(parsed),
        "parser_errors": parser_errors,
    }
    if not parsed:
        build_log["messages"] = logger.messages
        log_path.write_text(json.dumps(build_log, indent=2) + "\n")
        for e in parser_errors:
            print(f"PARSE ERROR {e['op']} {e['node']}: {e['desc']}")
        raise SystemExit(f"ONNX parse failed; see {log_path}")
    print(f"Parsed in {time.time() - t0:.0f}s: {network.num_layers} network layers")

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if args.opt_level is not None:
        config.builder_optimization_level = args.opt_level
    if args.no_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    tf32 = bool(config.get_flag(trt.BuilderFlag.TF32))

    print("Building engine ...", flush=True)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.time() - t0
    build_log["built"] = serialized is not None
    build_log["build_seconds"] = round(build_seconds, 1)
    build_log["messages"] = logger.messages
    log_path.write_text(json.dumps(build_log, indent=2) + "\n")
    if serialized is None:
        raise SystemExit(f"Engine build failed; see {log_path}")

    engine_path.write_bytes(bytes(serialized))
    print(f"Built in {build_seconds:.0f}s -> {engine_path} ({engine_path.stat().st_size / 1e6:.0f} MB)")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    io = {
        engine.get_tensor_name(i): {
            "mode": engine.get_tensor_mode(engine.get_tensor_name(i)).name,
            "dtype": engine.get_tensor_dtype(engine.get_tensor_name(i)).name,
            "shape": list(engine.get_tensor_shape(engine.get_tensor_name(i))),
        }
        for i in range(engine.num_io_tensors)
    }
    write_provenance(
        args.onnx.with_name(f"{name}.engine.provenance.json"),
        {
            "graph": onnx_prov["graph"],
            "engine": {
                "file": engine_path.name,
                "sha256": sha256_file(engine_path),
                "bytes": engine_path.stat().st_size,
                "layers": engine.num_layers,
                "device_memory_bytes": engine.device_memory_size_v2,
                "io": io,
            },
            "tensorrt": {
                "version": trt.__version__,
                "strongly_typed": True,
                "tf32": tf32,
                "builder_optimization_level": config.builder_optimization_level,
                "build_seconds": round(build_seconds, 1),
            },
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "cuda": torch.version.cuda,
            },
            # Engines are tied to this GPU and TensorRT version; rebuild elsewhere.
            "portable": False,
            "source_onnx": {
                "files": onnx_prov["files"],
                "opset": onnx_prov["opset"],
                "precision": onnx_prov["precision"],
                "checkpoints": onnx_prov["checkpoints"],
                "exporter": onnx_prov["exporter"],
            },
            "unsupported_ops": unsupported,
            "builder_warnings": len([m for m in logger.messages if m["severity"] == "WARNING"]),
            "builder_errors": len(
                [m for m in logger.messages if m["severity"] in ("ERROR", "INTERNAL_ERROR")]
            ),
        },
    )
    print(f"Build log -> {log_path}")


if __name__ == "__main__":
    main()
