"""Figures for deployment/README.md, drawn from the JSON results.

    python deployment/make_figures.py                                # full pipeline
    python deployment/make_figures.py --coarse-size 512 --refine-size none \
        --direction ab --refiners off                                # coarse only

One set of figures per pipeline configuration (pipeline.py), written as
``figures/<kind>__<config>.png``. Colour encodes compute precision (three
categorical slots); backends are identified by row labels, never by colour
alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PipelineConfig  # noqa: E402

RESULTS = Path("deployment/results")
FIGURES = Path("deployment/figures")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"fp32": "#2a78d6", "bf16": "#eb6834", "fp16": "#1baf7a", "ref": "#52514e"}
PRECISION_LABEL = {
    "fp32": "fp32 compute",
    "bf16": "bf16 compute (the authors' setting)",
    "fp16": "fp16 encoder",
}
# (label, precision group), in display order.
VARIANTS = {
    "torch_bf16": ("PyTorch original code, bf16", "bf16"),
    "torchtrt_bf16": ("Torch-TensorRT (partly PyTorch), bf16", "bf16"),
    "trt_bf16": ("ONNX export on TensorRT, bf16", "bf16"),
    "trt_fp16enc_bf16": ("ONNX export on TensorRT, fp16 encoder + bf16", "fp16"),
    "torch_fp32": ("PyTorch original code, fp32", "fp32"),
    "torchtrt_fp32": ("Torch-TensorRT, fp32", "fp32"),
    "onnx_cpu_fp32": ("ONNX export on ONNX Runtime CPU, fp32", "fp32"),
    "onnx_gpu_fp32": ("ONNX export on ONNX Runtime GPU, fp32", "fp32"),
    "trt_fp32": ("ONNX export on TensorRT, fp32", "fp32"),
    "trt_fp32_tf32": ("ONNX export on TensorRT, fp32 + TF32", "fp32"),
}
DATASETS = {
    "wxbs": "WxBS\nmAA@10px, 32 pairs",
    "hardmatch": "HardMatch\nmAA@10px",
    "diode": "DIODE RGB-depth\nAUC@10px, 892 pairs",
    "mmim_remote": "MMIM remote sensing\nAUC@10px, 41 pairs",
    "mmim_medical": "MMIM medical\nAUC@10px, 73 pairs",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend(fig, groups):
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, color=COLORS[g], label=PRECISION_LABEL[g])
        for g in groups
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), frameon=False,
        fontsize=9, labelcolor=INK2, handletextpad=0.3, columnspacing=1.6,
    )


def load_accuracy(tag: str) -> dict:
    out = {}
    for path in sorted((RESULTS / "datasets").glob(f"*__{tag}__*.json")):
        r = json.loads(path.read_text())
        if r["config"] == tag and "__max" not in path.name:
            out[(r["dataset"], r["variant"])] = r
    return out


def describe(cfg: PipelineConfig) -> str:
    text = f"{cfg.coarse[0]} px coarse"
    if not cfg.refiners:
        text += ", no refinement"
    elif cfg.refine is not None:
        text += f", refined at {cfg.refine[0]} px"
    else:
        text += f", refined at {cfg.coarse[0]} px"
    return text + (", both directions" if cfg.bidirectional else ", one direction")


def fig_accuracy(acc: dict, cfg: PipelineConfig) -> None:
    datasets = [d for d in DATASETS if any(k[0] == d for k in acc)]
    rows = [v for v in VARIANTS if any(k[1] == v for k in acc)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets) + 2.6, 0.36 * len(rows) + 1.9), sharey=True)
    axes = [axes] if len(datasets) == 1 else list(axes)
    for ax, d in zip(axes, datasets):
        style(ax)
        xs = []
        for i, v in enumerate(rows):
            r = acc.get((d, v))
            if r is None:
                continue
            c = COLORS[VARIANTS[v][1]]
            y = len(rows) - 1 - i
            if len(r["runs"]) > 1:
                ax.plot([r["min"], r["max"]], [y, y], color=c, linewidth=2, solid_capstyle="round", alpha=0.45)
            ax.plot(r["mean"], y, "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{r['mean']:.1f}", (r["max"], y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
            xs += [r["min"], r["max"]]
        if xs:
            pad = max(1.0, 0.12 * (max(xs) - min(xs)))
            ax.set_xlim(min(xs) - pad, max(xs) + 2.6 * pad)
        ax.set_title(DATASETS[d], fontsize=9, color=INK, loc="left")
        ax.set_ylim(-0.6, len(rows) - 0.4)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([VARIANTS[v][0] for v in reversed(rows)])
    fig.suptitle(
        f"Match quality by backend and precision ({describe(cfg)}; higher is better)",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.text(0.01, 0.085, "Dot = mean over seeds, bar = min-max over seeds (single-seed runs have no bar). "
             "Each panel has its own scale.", fontsize=8, color=INK2)
    legend(fig, [g for g in ("bf16", "fp16", "fp32") if any(VARIANTS[v][1] == g for v in rows)])
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(FIGURES / f"accuracy__{cfg.tag}.png", dpi=170)
    plt.close(fig)


def hbar(ax, labels, values, groups, fmt, notes=None):
    style(ax)
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, values, height=0.62, color=[COLORS[g] for g in groups], edgecolor=SURFACE, linewidth=1.5)
    for i, (y, v) in enumerate(zip(ys, values)):
        text = fmt(v) + (f"  {notes[i]}" if notes and notes[i] else "")
        ax.annotate(text, (v, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(values) * 1.22)


def fig_runtime(rt: dict, cfg: PipelineConfig) -> None:
    recs = {r["variant"]: r for r in rt["variants"] if "error" not in r}
    lat_rows = [v for v in VARIANTS if v in recs and v != "onnx_cpu_fp32"]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(lat_rows) + 1.7))
    hbar(
        ax,
        [VARIANTS[v][0] for v in lat_rows],
        [recs[v]["total"]["median_ms"] for v in lat_rows],
        [VARIANTS[v][1] for v in lat_rows],
        lambda v: f"{v:.0f} ms",
    )
    ax.set_xlabel("median latency per image pair, ms (lower is better)")
    fig.suptitle(f"Latency at batch 1 ({describe(cfg)}), NVIDIA GB10",
                 x=0.01, ha="left", fontsize=11, color=INK)
    legend(fig, ["bf16", "fp16", "fp32"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(FIGURES / f"latency__{cfg.tag}.png", dpi=170)
    plt.close(fig)

    # Memory ladder: Torch-TensorRT is left out (it compiles in-process, so its
    # peak is a build footprint, not a deployment one).
    mem_rows = [v for v in VARIANTS if v in recs and not v.startswith("torchtrt")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 0.42 * len(mem_rows) + 1.8), sharey=True)
    labels = [VARIANTS[v][0] for v in mem_rows]
    groups = [VARIANTS[v][1] for v in mem_rows]
    hbar(axes[0], labels, [recs[v]["system_peak_mb"] / 1024 for v in mem_rows], groups, lambda v: f"{v:.1f} GB")
    axes[0].set_title("Peak memory while running (unified host + GPU)", fontsize=9, color=INK, loc="left")
    axes[0].set_xlabel("GB (lower is better)")
    hbar(axes[1], labels, [recs[v]["disk_mb"] / 1024 for v in mem_rows], groups, lambda v: f"{v:.1f} GB")
    axes[1].set_title("Weights loaded from disk", fontsize=9, color=INK, loc="left")
    axes[1].set_xlabel("GB")
    fig.suptitle(f"Memory ladder at batch 1 ({describe(cfg)})", x=0.01, ha="left", fontsize=11, color=INK)
    legend(fig, ["bf16", "fp16", "fp32"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(FIGURES / f"memory__{cfg.tag}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    PipelineConfig.add_arguments(p)
    cfg = PipelineConfig.from_args(p.parse_args())
    FIGURES.mkdir(parents=True, exist_ok=True)
    acc = load_accuracy(cfg.tag)
    if acc:
        fig_accuracy(acc, cfg)
    rt_path = RESULTS / f"runtime__{cfg.tag}.json"
    if rt_path.exists():
        fig_runtime(json.loads(rt_path.read_text()), cfg)
    print(f"Figures for {cfg.tag} -> {FIGURES}")


if __name__ == "__main__":
    main()
