"""Pipeline configuration shared by the export, build and evaluation scripts.

A RoMa-Omega pipeline is a chain of static-shape graphs:

    encoder   (img_A, img_B at the coarse size)        -> VGGT-Omega features
    matcher   (features)                               -> coarse warp + overlap
    refiner   (coarse-size images, coarse warp)        -> warp at the coarse size
    refiner2  (refine-size images, first-pass warp)    -> warp at the refine size

The defaults reproduce upstream's ``precise`` setting (the released model's
default): 800 px coarse stage, both matching directions, refiners on, second
refinement pass at 1280 px. Every part is configurable:

    --coarse-size 800        coarse resolution, ``N`` or ``HxW`` (multiples of 16)
    --refine-size 1280       second refinement pass, ``N``, ``HxW`` or ``none``
    --direction both         ``both`` (A->B and B->A) or ``ab`` (A->B only)
    --refiners on            ``off`` stops after the coarse matcher

Upstream's presets expressed this way: ``precise`` is the default; ``base`` is
``--coarse-size 640 --refine-size none --direction ab``; ``fast`` is 512 and
``turbo`` 320 with the same flags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

PATCH_SIZE = 16
PRECISIONS = ("fp32", "bf16", "fp16")
DIRECTIONS = ("AB", "BA")


def _parse_size(text: str) -> tuple[int, int] | None:
    if text.lower() in ("none", "off", "0"):
        return None
    h, _, w = text.lower().partition("x")
    return int(h), int(w or h)


@dataclass(frozen=True)
class PipelineConfig:
    coarse: tuple[int, int] = (800, 800)
    refine: tuple[int, int] | None = (1280, 1280)
    bidirectional: bool = True
    refiners: bool = True

    def __post_init__(self):
        assert self.coarse[0] % PATCH_SIZE == 0 and self.coarse[1] % PATCH_SIZE == 0, (
            f"coarse size must be a multiple of {PATCH_SIZE}: {self.coarse}"
        )
        if not self.refiners:
            assert self.refine is None, "a second refinement pass needs --refiners on"

    # -- naming -------------------------------------------------------------
    @property
    def tag(self) -> str:
        """Short name used in result and figure file names."""
        t = f"c{self.coarse[0]}x{self.coarse[1]}"
        if self.refine is not None:
            t += f"_r{self.refine[0]}x{self.refine[1]}"
        t += "_bidir" if self.bidirectional else "_ab"
        if not self.refiners:
            t += "_coarse"
        return t

    @property
    def directions(self) -> tuple[str, ...]:
        return DIRECTIONS if self.bidirectional else DIRECTIONS[:1]

    @property
    def stages(self) -> tuple[str, ...]:
        """Graph names, in execution order."""
        names = ["encoder", "matcher"]
        if self.refiners:
            names.append("refiner")
            if self.refine is not None:
                names.append("refiner2")
        return tuple(names)

    def graph_dir(self, stage: str, precision: str, artifacts: Path) -> Path:
        ch, cw = self.coarse
        bidir = "_bidir" if self.bidirectional else ""
        if stage == "encoder":
            name = f"encoder_{ch}x{cw}_{precision}"
        elif stage == "matcher":
            name = f"matcher_{ch}x{cw}_{precision}{bidir}"
        elif stage == "refiner":
            name = f"refiner_{ch}x{cw}_{precision}{bidir}"
        elif stage == "refiner2":
            rh, rw = self.refine
            name = f"refiner_{rh}x{rw}_from{ch}x{cw}_{precision}{bidir}"
        else:
            raise ValueError(stage)
        return Path(artifacts) / name

    def graph_path(self, stage: str, precision: str, artifacts: Path) -> Path:
        stem = "refiner" if stage == "refiner2" else stage
        return self.graph_dir(stage, precision, artifacts) / f"{stem}.onnx"

    def describe(self) -> dict:
        return {
            "tag": self.tag,
            "coarse_size": list(self.coarse),
            "refine_size": list(self.refine) if self.refine else None,
            "direction": "both" if self.bidirectional else "ab",
            "refiners": self.refiners,
        }

    # -- command line -------------------------------------------------------
    @staticmethod
    def add_arguments(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("pipeline (defaults = upstream 'precise')")
        g.add_argument("--coarse-size", default="800", help="coarse resolution, N or HxW")
        g.add_argument("--refine-size", default="1280", help="second refinement pass: N, HxW or none")
        g.add_argument("--direction", choices=["both", "ab"], default="both")
        g.add_argument("--refiners", choices=["on", "off"], default="on")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        refiners = args.refiners == "on"
        return cls(
            coarse=_parse_size(args.coarse_size),
            refine=_parse_size(args.refine_size) if refiners else None,
            bidirectional=args.direction == "both",
            refiners=refiners,
        )

    def to_args(self) -> list[str]:
        """The command-line flags that reproduce this configuration."""
        return [
            "--coarse-size", f"{self.coarse[0]}x{self.coarse[1]}",
            "--refine-size", f"{self.refine[0]}x{self.refine[1]}" if self.refine else "none",
            "--direction", "both" if self.bidirectional else "ab",
            "--refiners", "on" if self.refiners else "off",
        ]
