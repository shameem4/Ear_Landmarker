"""Retrain the v1-equivalent model for attribution purposes only.

Why this exists: v1's published val NME (0.0307) cannot be compared against v2.
It was measured on the old two-way split, whose training set overlaps the new
held-out test set, and it was also the early-stopping / checkpoint-selection
metric. To say what fixing the rotation bug was actually worth, v1's *behaviour*
has to be re-measured on the new splits under the new protocol.

This script re-enables the v1 rotation-label bug and nothing else, so the only
difference from a v2 run is the bug itself.

Do not use this for anything but that one comparison.

Usage:
    python scripts/train_baseline_v1.py --epochs 500
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.dataset as dataset_mod  # noqa: E402


def main() -> None:
    # Re-enable the v1 label bug before the dataset is constructed.
    dataset_mod.LEGACY_ROTATION_SIGN = True
    print("=" * 72)
    print("BASELINE RUN -- v1 rotation-label bug DELIBERATELY ENABLED.")
    print("Rotated samples are mislabelled by ~12.5px (mean, at 192px).")
    print("This exists only to attribute the v2 improvement. Not a real model.")
    print("=" * 72)

    import train  # noqa: E402  (imported after the patch)

    # Default the run name so baseline artefacts never overwrite a v2 run.
    if not any(a.startswith("--run-name") for a in sys.argv[1:]):
        sys.argv += ["--run-name", "baseline_v1"]

    train.main()


if __name__ == "__main__":
    main()
