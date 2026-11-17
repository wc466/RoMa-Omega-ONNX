"""Validate the exported matcher graph, alone and chained after the encoder graph.

The reference is upstream ``Matcher.forward`` (unmodified), run two ways:

  * fp32          - autocast disabled; what the graphs were exported from
  * bf16 autocast - how upstream RoMa-Omega actually runs on CUDA

Three ONNX results are compared against those references:

  * matcher only  - matcher.onnx fed with PyTorch fp32 encoder features
  * chained       - encoder.onnx -> matcher.onnx, images in, warp out

Warp error is reported in pixels of the input image (the warp is a normalised
coordinate in the other view), overall and on confidently overlapping pixels.

Run from the repository root:

    python deployment/validate_matcher.py \
        --encoder deployment/artifacts/encoder_512x512_fp32/encoder.onnx \
        --matcher deployment/artifacts/matcher_512x512_fp32/matcher.onnx
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
from common import (  # noqa: E402
    EncoderWrapper,
    load_image,
    load_roma_omega,
    no_autocast,
    set_matcher_amp,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compare(test: dict, ref: dict, H: int, W: int, direction: str) -> dict:
    """Compare warp/confidence dicts of numpy arrays for one direction."""
    warp_t = test[f"warp_{direction}"].astype(np.float64)
    warp_r = ref[f"warp_{direction}"].astype(np.float64)
    conf_t = sigmoid(test[f"confidence_{direction}"].astype(np.float64))[..., 0]
    conf_r = sigmoid(ref[f"confidence_{direction}"].astype(np.float64))[..., 0]
    # Normalised [-1, 1] -> pixels of the (H, W) input image.
    scale = np.array([W / 2.0, H / 2.0])
    err_px = np.linalg.norm((warp_t - warp_r) * scale, axis=-1)
    confident = conf_r > 0.5
    out = {
        "warp_px_mean": float(err_px.mean()),
        "warp_px_max": float(err_px.max()),
        "confident_fraction": float(confident.mean()),
        "overlap_abs_mean": float(np.abs(conf_t - conf_r).mean()),
        "overlap_abs_max": float(np.abs(conf_t - conf_r).max()),
    }
    if confident.any():
        e = err_px[confident]
        out.update(
            {
                "warp_px_mean_confident": float(e.mean()),
                "warp_px_p99_confident": float(np.percentile(e, 99)),
                "warp_px_max_confident": float(e.max()),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--matcher", type=Path, required=True)
    p.add_argument("--img-a", default=".WxBS/v1.1/WGALBS/bridge/01.png")
    p.add_argument("--img-b", default=".WxBS/v1.1/WGALBS/bridge/02.png")
    p.add_argument("--provider", default="CPUExecutionProvider")
    args = p.parse_args()

    prov = json.loads(args.matcher.with_name("matcher.provenance.json").read_text())
    H, W = prov["image_size"]
    bidirectional = prov["bidirectional"]
    directions = ["AB", "BA"] if bidirectional else ["AB"]
    feat_names = list(prov["inputs"].keys())
    out_names = list(prov["outputs"].keys())

    rng = np.random.default_rng(0)
    pairs = {
        "real": (load_image(args.img_a, H, W), load_image(args.img_b, H, W)),
        "random": (
            rng.random((1, 3, H, W), dtype=np.float32),
            rng.random((1, 3, H, W), dtype=np.float32),
        ),
    }

    print(f"ONNX Runtime {ort.__version__}, provider {args.provider}")
    enc_sess = ort.InferenceSession(str(args.encoder), providers=[args.provider])
    mat_sess = ort.InferenceSession(str(args.matcher), providers=[args.provider])
    assert [o.name for o in enc_sess.get_outputs()] == feat_names, "encoder/matcher layer mismatch"

    model = load_roma_omega()
    matcher = model.matcher
    dev = next(matcher.parameters()).device
    encoder = EncoderWrapper(matcher.vggt.aggregator, tuple(prov["cached_layers"])).eval()

    def torch_matcher(ta, tb) -> dict:
        preds = matcher(ta, tb, bidirectional=bidirectional)
        return {
            n: preds[n].float().cpu().numpy() for n in out_names
        }

    report = {
        "encoder": str(args.encoder),
        "matcher": str(args.matcher),
        "provider": args.provider,
        "image_size": [H, W],
        "pairs": {},
    }
    for pair_name, (a, b) in pairs.items():
        ta, tb = torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)
        with torch.inference_mode():
            # Upstream as shipped: bf16 autocast in matcher and DPT head.
            set_matcher_amp(matcher, True)
            ref_bf16 = torch_matcher(ta, tb)
            # Pure fp32 reference through the same unmodified forward.
            set_matcher_amp(matcher, False)
            with no_autocast():
                ref_fp32 = torch_matcher(ta, tb)
                feats_fp32 = [f.cpu().numpy() for f in encoder(ta, tb)]
            set_matcher_amp(matcher, True)

        t0 = time.time()
        only = dict(zip(out_names, mat_sess.run(None, dict(zip(feat_names, feats_fp32)))))
        t_matcher = time.time() - t0

        t0 = time.time()
        feats_onnx = enc_sess.run(None, {"img_A": a, "img_B": b})
        t_encoder = time.time() - t0
        chained = dict(zip(out_names, mat_sess.run(None, dict(zip(feat_names, feats_onnx)))))

        res = {
            "onnx_encoder_seconds": round(t_encoder, 2),
            "onnx_matcher_seconds": round(t_matcher, 2),
        }
        for d in directions:
            res[d] = {
                "matcher_onnx_vs_torch_fp32": compare(only, ref_fp32, H, W, d),
                "chained_onnx_vs_torch_fp32": compare(chained, ref_fp32, H, W, d),
                "chained_onnx_vs_torch_bf16": compare(chained, ref_bf16, H, W, d),
                # How far upstream's own bf16 path is from fp32, for context.
                "torch_bf16_vs_torch_fp32": compare(ref_bf16, ref_fp32, H, W, d),
            }
        report["pairs"][pair_name] = res

    out_path = args.matcher.with_name("matcher.validation.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    for pair_name, res in report["pairs"].items():
        print(
            f"\n[{pair_name}] onnx encoder {res['onnx_encoder_seconds']}s, "
            f"matcher {res['onnx_matcher_seconds']}s"
        )
        for d in directions:
            for key, m in res[d].items():
                conf = (
                    f"confident: mean={m['warp_px_mean_confident']:.4f} "
                    f"p99={m['warp_px_p99_confident']:.4f} max={m['warp_px_max_confident']:.4f}"
                    if "warp_px_mean_confident" in m
                    else "confident: none"
                )
                print(
                    f"  {d} {key:28s} warp px mean={m['warp_px_mean']:.4f} "
                    f"max={m['warp_px_max']:.3f} | {conf} "
                    f"({m['confident_fraction']:.0%}) | overlap max diff={m['overlap_abs_max']:.4f}"
                )
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
