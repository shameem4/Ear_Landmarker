"""Score the backbone-pretraining experiment against its pre-registered rule.

Safe to run while the experiment is still going: it reports progress and says
the result is not final until every run has logged a test NME.

The rule (fixed before the runs started, see run_pretrain_experiment.sh):
  a win needs BOTH complete rank separation across 3 seeds AND a mean
  improvement above 1% relative. Anything less is a seed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "runs" / "logs"
SEEDS = (42, 1, 2)
MIN_RELATIVE_GAIN = 0.01          # the seed-noise floor on the shipped model


def read(run: str) -> dict:
    f = LOGS / run / "version_0" / "metrics.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(f.open()))
    val = [float(r["val_nme"]) for r in rows if r.get("val_nme", "")]
    test = [float(r["test/nme"]) for r in rows if r.get("test/nme", "")]
    epochs = [int(float(r["epoch"])) for r in rows if r.get("epoch", "")]
    return {"val_best": min(val) if val else None,
            "test": test[-1] if test else None,
            "epoch": max(epochs) if epochs else 0}


def main() -> None:
    arms = {"control": [], "transfer": []}
    print(f"{'run':22s} {'epoch':>6} {'val best':>9} {'test':>9}")
    for arm in arms:
        for s in SEEDS:
            run = f"pre_{arm}_s{s}"
            r = read(run)
            if not r:
                print(f"{run:22s} {'-':>6} {'not started':>9}")
                continue
            print(f"{run:22s} {r['epoch']:6d} "
                  f"{r['val_best']:9.5f} " if r['val_best'] else f"{run:22s} {r['epoch']:6d} {'-':>9} ",
                  end="")
            print(f"{r['test']:9.5f}" if r["test"] is not None else f"{'running':>9}")
            arms[arm].append(r)

    done = {k: [r["test"] for r in v if r["test"] is not None] for k, v in arms.items()}
    if len(done["control"]) < len(SEEDS) or len(done["transfer"]) < len(SEEDS):
        print(f"\nnot final: {len(done['control'])}/{len(SEEDS)} control and "
              f"{len(done['transfer'])}/{len(SEEDS)} transfer runs have a test NME")
        return

    c, t = np.array(done["control"]), np.array(done["transfer"])
    gain = (c.mean() - t.mean()) / c.mean()
    separated = t.max() < c.min()

    print(f"\ncontrol  {c.mean():.5f} +/- {c.std(ddof=1):.5f}   {np.sort(c)}")
    print(f"transfer {t.mean():.5f} +/- {t.std(ddof=1):.5f}   {np.sort(t)}")
    print(f"\nrelative change: {gain*100:+.2f}%  (positive = transfer better)")
    print(f"complete rank separation: {'yes' if separated else 'no'}")

    print("\nPRE-REGISTERED VERDICT:")
    if separated and gain > MIN_RELATIVE_GAIN:
        print("  WIN -- transfer helps. Both conditions met.")
        print("  Next: this was measured on the 82K mirrored backbone, so porting")
        print("  the result to the shipped 340K architecture needs a donor trained")
        print("  at that width. That is real work; the result justifies starting it.")
    elif separated and gain < -MIN_RELATIVE_GAIN:
        print("  LOSS -- transfer hurts, at the same strength. Question closed.")
    else:
        print("  NOT SEPARABLE -- this is a seed, not a result.")
        print("  Do not report it as a small improvement; the pre-registered rule")
        print("  says anything under 1% with incomplete separation is noise.")
        print("  Backbone pretraining from this donor does not help; FaceMesh")
        print("  transfer, which shares even less structure, is unlikely to.")


if __name__ == "__main__":
    main()
