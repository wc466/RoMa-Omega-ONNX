"""Dataset harness: run any matcher variant on any benchmark, across seeds.

Datasets
  wxbs          WxBS v1.1 test (32 pairs), mAA@10px           [repo benchmark]
  hardmatch     HardMatch test, mAA@10px                      [repo benchmark]
  diode         DIODE val outdoor RGB-depth, homography AUC   [MINIMA protocol]
  mmim_medical  MMIM medical, homography AUC                  [MINIMA protocol]
  mmim_remote   MMIM remote sensing, homography AUC           [MINIMA protocol]

Variants: see matchers.py. The seed affects match sampling and the robust
estimator (MAGSAC / RANSAC), not the network.

Run from the repository root:

    python deployment/eval_datasets.py --datasets wxbs,hardmatch \
        --variants torch_bf16_full --seeds 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matchers import VARIANTS, RoMaOmegaVariants  # noqa: E402

DATASETS = ("wxbs", "hardmatch", "diode", "mmim_medical", "mmim_remote")
# Headline number per dataset, as (key path, label).
HEADLINE = {
    "wxbs": "mAA_10px",
    "hardmatch": "mAA_10px",
    "diode": "auc@10",
    "mmim_medical": "auc@10",
    "mmim_remote": "auc@10",
}


def build_benchmark(name: str, args):
    if name == "wxbs":
        from romaomega.benchmarks.general import WxBSBenchmark

        return WxBSBenchmark()
    if name == "hardmatch":
        from hardmatch import HardMatchBenchmark

        return HardMatchBenchmark()
    from benchmarks_minima import DiodeBenchmark, MmimBenchmark

    if name == "diode":
        return DiodeBenchmark(args.diode_root, args.minima_root, args.max_pairs)
    return MmimBenchmark(args.mmim_root, name)


def headline(dataset: str, result: dict) -> float:
    key = HEADLINE[dataset]
    if key in result:
        return 100 * result[key]  # repo benchmarks report fractions
    return result["mean_over_scenes"][key]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", default="wxbs")
    p.add_argument("--variants", default="torch_bf16_full")
    p.add_argument("--seeds", default="1")
    p.add_argument("--artifacts", type=Path, default=Path("deployment/artifacts"))
    p.add_argument("--num-corresp", type=int, default=None)
    p.add_argument("--diode-root", type=Path, default=Path("data/DIODE"))
    p.add_argument("--mmim-root", type=Path, default=Path("data/MMIM"))
    p.add_argument("--minima-root", type=Path, default=Path("MINIMA"))
    p.add_argument("--max-pairs", type=int, default=None, help="DIODE only: limit pairs (smoke tests)")
    p.add_argument("--out-dir", type=Path, default=Path("deployment/results/datasets"))
    args = p.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    variants = [v for v in args.variants.split(",") if v]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    assert all(d in DATASETS for d in datasets), datasets
    assert all(v in VARIANTS for v in variants), variants

    from romaomega.random import set_seed

    Path("data").mkdir(exist_ok=True)
    # Benchmarks are built before seeding: DIODE homographies are seeded per file.
    benchmarks = {d: build_benchmark(d, args) for d in datasets}
    models = RoMaOmegaVariants(args.artifacts, num_corresp=args.num_corresp)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for dataset in datasets:
        for variant in variants:
            matcher = models.select(variant)
            runs = []
            for seed in seeds:
                set_seed(seed)
                t0 = time.time()
                empty_before = models.empty_matches
                result = benchmarks[dataset].benchmark(matcher)
                runs.append(
                    {
                        "seed": seed,
                        "headline": headline(dataset, result),
                        "seconds": round(time.time() - t0, 1),
                        "empty_matches": models.empty_matches - empty_before,
                        "result": result,
                    }
                )
                print(
                    f"== {dataset} {variant} seed {seed}: "
                    f"{HEADLINE[dataset]} {runs[-1]['headline']:.2f} ({runs[-1]['seconds']}s)",
                    flush=True,
                )
            scores = [r["headline"] for r in runs]
            record = {
                "dataset": dataset,
                "variant": variant,
                "metric": HEADLINE[dataset],
                "scale": "0-100",
                "setting": models.describe(variant),
                "seeds": seeds,
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else None,
                "min": min(scores),
                "max": max(scores),
                "runs": runs,
            }
            suffix = "" if args.max_pairs is None else f"__max{args.max_pairs}"
            (args.out_dir / f"{dataset}__{variant}{suffix}.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            summary.append(record)

    print(f"\n{'dataset':14s} {'variant':20s} {'metric':9s} {'mean':>7s} {'std':>6s} {'min':>7s} {'max':>7s}  runs")
    for r in summary:
        std = f"{r['std']:.2f}" if r["std"] is not None else "   -"
        print(
            f"{r['dataset']:14s} {r['variant']:20s} {r['metric']:9s} {r['mean']:7.2f} {std:>6s} "
            f"{r['min']:7.2f} {r['max']:7.2f}  {len(r['runs'])}"
        )
    if models.reduced_count_matches:
        print(f"note: {models.reduced_count_matches} match calls sampled fewer matches than requested")
    print(f"Results -> {args.out_dir}")


if __name__ == "__main__":
    main()
