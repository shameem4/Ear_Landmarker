"""Score the FaceMesh-copy experiment against its pre-registered rule.

Safe to run mid-experiment; it says when the result is not yet final.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "runs" / "logs"
SEEDS = (42, 1, 2)

# The control, already run. Three seeds of v6_persp65, test NME.
CONTROL = np.array([0.02923, 0.02948, 0.02886])
TWO_SIGMA = 2 * CONTROL.std(ddof=1)


def read(run: str) -> dict:
    f = LOGS / run / "version_0" / "metrics.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(f.open()))
    val = [float(r["val_nme"]) for r in rows if r.get("val_nme", "")]
    test = [float(r["test/nme"]) for r in rows if r.get("test/nme", "")]
    ep = [int(float(r["epoch"])) for r in rows if r.get("epoch", "")]
    return {"val": min(val) if val else None,
            "test": test[-1] if test else None,
            "epoch": max(ep) if ep else 0}


def main() -> None:
    arms = {"fm_scratch": [], "fm_pre": []}
    print(f"{'run':20s} {'epoch':>6} {'val best':>9} {'test':>9}")
    for arm in arms:
        for s in SEEDS:
            r = read(f"{arm}_s{s}")
            if not r:
                print(f"{arm}_s{s:<14} {'-':>6} {'not started':>9}")
                continue
            v = f"{r['val']:9.5f}" if r["val"] else f"{'-':>9}"
            t = f"{r['test']:9.5f}" if r["test"] is not None else f"{'running':>9}"
            print(f"{arm}_s{s:<14} {r['epoch']:6d} {v} {t}")
            arms[arm].append(r)

    done = {k: np.array([r["test"] for r in v if r["test"] is not None])
            for k, v in arms.items()}
    print(f"\ncontrol (v6_persp65, already run)  {CONTROL.mean():.5f} "
          f"+/- {CONTROL.std(ddof=1):.5f}   {np.sort(CONTROL)}")

    if any(len(v) < len(SEEDS) for v in done.values()):
        print("\nnot final: " + ", ".join(
            f"{k} {len(v)}/{len(SEEDS)}" for k, v in done.items()))
        return

    for k, v in done.items():
        print(f"{k:34s} {v.mean():.5f} +/- {v.std(ddof=1):.5f}   {np.sort(v)}")

    print("\nPRE-REGISTERED VERDICT")
    for k, v in done.items():
        d = v.mean() - CONTROL.mean()
        if v.mean() < CONTROL.mean() - TWO_SIGMA and v.max() < CONTROL.min():
            verdict = "BEATS EarLandmarker"
        elif abs(d) <= TWO_SIGMA:
            verdict = "MATCHES EarLandmarker (within 2 sigma; not a win either way)"
        elif d > TWO_SIGMA:
            verdict = "LOSES to EarLandmarker"
        else:
            verdict = "better on the mean but seeds overlap -- not separable"
        print(f"  {k:12s} {d:+.5f} vs control   {verdict}")

    s, p = done["fm_scratch"], done["fm_pre"]
    gain = (s.mean() - p.mean()) / s.mean()
    sep = p.max() < s.min()
    print(f"\n  pretraining effect: {gain*100:+.2f}% "
          f"(rank separation: {'yes' if sep else 'no'})")
    print("  " + ("pretrained weights help" if sep and gain > 0.01 else
                  "pretrained weights hurt" if sep and gain < -0.01 else
                  "not separable -- the free weights do not measurably matter"))


if __name__ == "__main__":
    main()
