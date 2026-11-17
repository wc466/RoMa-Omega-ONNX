"""Validate an exported encoder graph against PyTorch.

Runs the same image pair through ONNX Runtime and through the PyTorch encoder
and reports per-output error. Two PyTorch references are used:

  * fp32          - what the graph was exported from (should agree tightly)
  * bf16 autocast - how upstream RoMa-Omega actually runs the encoder on CUDA

Run from the repository root:

    python deployment/validate_encoder.py \
        --onnx deployment/artifacts/encoder_512x512_fp32/encoder.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_encoder, load_image  # noqa: E402


def compare(test: np.ndarray, ref: np.ndarray) -> dict:
    test = test.astype(np.float64)
    ref = ref.astype(np.float64)
    diff = np.abs(test - ref)
    # Per-token cosine similarity over the feature dimension.
    cos = (test * ref).sum(-1) / (
        np.linalg.norm(test, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-12
    )
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l2": float(np.linalg.norm(test - ref) / np.linalg.norm(ref)),
        "min_cos": float(cos.min()),
        "mean_cos": float(cos.mean()),
        "ref_abs_mean": float(np.abs(ref).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--checkpoint", default="vggtomega.pt")
    p.add_argument("--img-a", default=".WxBS/v1.1/WGALBS/bridge/01.png")
    p.add_argument("--img-b", default=".WxBS/v1.1/WGALBS/bridge/02.png")
    p.add_argument("--provider", default="CPUExecutionProvider")
    args = p.parse_args()

    provenance = json.loads(
        args.onnx.with_name(args.onnx.stem + ".provenance.json").read_text()
    )
    layers = tuple(provenance["cached_layers"])
    _, _, H, W = provenance["inputs"]["img_A"]

    rng = np.random.default_rng(0)
    pairs = {
        "real": (load_image(args.img_a, H, W), load_image(args.img_b, H, W)),
        "random": (
            rng.random((1, 3, H, W), dtype=np.float32),
            rng.random((1, 3, H, W), dtype=np.float32),
        ),
    }

    print(f"ONNX Runtime {ort.__version__}, provider {args.provider}")
    session = ort.InferenceSession(str(args.onnx), providers=[args.provider])
    out_names = [o.name for o in session.get_outputs()]

    encoder = load_encoder(args.checkpoint, layers)
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        encoder = encoder.cuda()
    dev = "cuda" if has_cuda else "cpu"

    report = {"onnx": str(args.onnx), "provider": args.provider, "pairs": {}}
    for pair_name, (a, b) in pairs.items():
        t0 = time.time()
        onnx_out = session.run(None, {"img_A": a, "img_B": b})
        t_onnx = time.time() - t0

        ta, tb = torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)
        with torch.inference_mode():
            ref_fp32 = [o.float().cpu().numpy() for o in encoder(ta, tb)]
            if has_cuda:
                with torch.autocast("cuda", torch.bfloat16):
                    ref_bf16 = [o.float().cpu().numpy() for o in encoder(ta, tb)]

        res = {"onnx_seconds": round(t_onnx, 2), "outputs": {}}
        for i, name in enumerate(out_names):
            entry = {"shape": list(onnx_out[i].shape)}
            entry[f"onnx_vs_torch_fp32_{dev}"] = compare(onnx_out[i], ref_fp32[i])
            if has_cuda:
                entry["onnx_vs_torch_bf16_cuda"] = compare(onnx_out[i], ref_bf16[i])
                # How far upstream's own bf16 path is from fp32, for context.
                entry["torch_bf16_vs_torch_fp32_cuda"] = compare(ref_bf16[i], ref_fp32[i])
            res["outputs"][name] = entry
        report["pairs"][pair_name] = res

    out_path = args.onnx.with_name(args.onnx.stem + ".validation.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    for pair_name, res in report["pairs"].items():
        print(f"\n[{pair_name}] onnx run {res['onnx_seconds']}s")
        for name, entry in res["outputs"].items():
            print(f"  {name} {entry['shape']}")
            for key, m in entry.items():
                if key == "shape":
                    continue
                print(
                    f"    {key:32s} max_abs={m['max_abs']:.3e} "
                    f"rel_l2={m['rel_l2']:.3e} min_cos={m['min_cos']:.6f}"
                )
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
