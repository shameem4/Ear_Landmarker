"""Ingest Ear_3D_Lifter's hand-corrected renders (synthetic_v1) as a training source.

Source: Ear_3D_Lifter/runs/synthetic_v1/ (scripts/handfix/3_render_training.py).
68 ears from 54 subjects (SONICOM, HUTUBS), 40 views each, 384 px crops framed at
landmark extent / 0.70. All 55 points of every ear were hand-checked in 3D, so
every point is supervised, including self-occluded ones at their projected
position (the same convention as the photo labels, which extrapolate hidden
points). The 16 HUTUBS heads Ear_3D_Lifter holds out end to end are not in the
delivery. Licences per row in the source manifest; the renders are of scan
geometry, so data/synthetic_v1 is gitignored and never published.

SPLIT BY SUBJECT: every view and both ears of a head go to one split, since 40
views of one ear are near-duplicates. test.csv is for scoring only; train.py
never reads val/test from a synthetic source. train_25/50.csv are nested subject
subsets of train.csv, for a learning curve.

synthetic_v2 (2026-10-22) is v1 (68 attached-lobe ears, byte-identical) plus 51
free-lobe ears, one per subject (list: Ear_3D_Lifter/runs/handfix/
_batch2_free_lobes.txt; ~30 visibly free, ~20 ambiguous, not verified).
--carry-split keeps every subject of an earlier ingest in its old split, so the
earlier synthetic test set stays held out; only new subjects are shuffled, with
--test-subjects/--val-subjects counted among them.

Usage:
    python scripts/ingest_synthetic_v1.py                   # synthetic_v1
    python scripts/ingest_synthetic_v1.py --name synthetic_v2 \\
        --carry-split data/synthetic_v1 --test-subjects 9 --val-subjects 5
    python scripts/ingest_synthetic_v1.py --name synthetic_v3 --px 600 \\
        --test-subjects 22 --val-subjects 22

synthetic_v3 (2026-10-23) is a different kind of delivery: 444 ears x 4 skins x
25 orbit views, each the WHOLE 600 px pipeline render at occupancy 0.75, for the
render-only model the 3D path uses. synthetic_v3r replaced it (2026-10-23):
20% of v3 were broken SONICOM decimate/noise renders. synthetic_v3j is an
add-on of 140 training ears with skin-coloured jewellery; ingest it with
--carry-split data/synthetic_v3r --addon so every ear keeps its split. 210 ears carry the user's hand labels, 234
lm3d_v13 (model-derived; `trusted` column in the source manifest).
"""
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = {f"pp{n}" for n in (16, 19, 21, 22, 33, 40, 46, 47, 49, 57, 69, 70, 71, 72, 77, 78)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-subjects", type=int, default=10)
    ap.add_argument("--val-subjects", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="synthetic_v1")
    ap.add_argument("--px", type=float, default=384.0, help="render size the labels are in")
    ap.add_argument("--carry-split", type=Path, help="earlier ingest whose subject split to keep")
    ap.add_argument("--addon", action="store_true",
                    help="delivery is a SUBSET of --carry-split's subjects (e.g. synthetic_v3j)")
    a = ap.parse_args()
    SRC = ROOT.parent / "Ear_3D_Lifter" / "runs" / a.name
    OUT = ROOT / "data" / a.name
    rows = list(csv.DictReader(open(SRC / "manifest.csv")))
    bad = {r["subject"] for r in rows} & HELD_OUT
    assert not bad, f"held-out heads in the delivery: {bad}"
    pts = []
    for r in rows:
        x = np.loadtxt(SRC / f"{r['stem']}_annotation.txt")
        assert x.shape == (55, 2) and np.isfinite(x).all(), r["stem"]
        pts.append(x / a.px)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "landmarks.npy", np.stack(pts).astype(np.float32))
    subj = sorted({r["subject"] for r in rows})
    old = {}
    if a.carry_split:
        for sp in ("train", "val", "test"):
            old.update({r["subject"]: sp for r in csv.DictReader(open(a.carry_split / f"{sp}.csv"))})
        if not a.addon:
            assert set(old) <= set(subj), "earlier subjects missing from the delivery"
        else:
            assert set(subj) <= set(old), "an add-on must not bring new subjects"
    new = [s for s in subj if s not in old]
    random.Random(a.seed).shuffle(new)
    test = {s for s in old if old[s] == "test"} | set(new[:a.test_subjects])
    val = {s for s in old if old[s] == "val"} | set(new[a.test_subjects:a.test_subjects + a.val_subjects])
    train = [s for s in subj if old.get(s) == "train"] + new[a.test_subjects + a.val_subjects:]
    fields = ["idx", "image_file", "source", "subject", "side", "yaw", "pitch"]

    def write(name, keep):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); n = 0
            for i, r in enumerate(rows):
                if keep(r["subject"]):
                    w.writerow({"idx": i, "image_file": f"images/{r['stem']}.png", "source": r["dataset"],
                                "subject": r["subject"], "side": r["side"], "yaw": r["yaw"], "pitch": r["pitch"]})
                    n += 1
        return n

    counts = {"manifest.csv": write("manifest.csv", lambda s: True),
              "train.csv": write("train.csv", lambda s: s in set(train)),
              "val.csv": write("val.csv", lambda s: s in val),
              "test.csv": write("test.csv", lambda s: s in test)}
    for frac in (25, 50):                                   # nested subsets
        keep = set(train[:max(1, round(len(train) * frac / 100))])
        counts[f"train_{frac}.csv"] = write(f"train_{frac}.csv", lambda s, k=keep: s in k)
    for r in rows:
        dst = OUT / "images" / f"{r['stem']}.png"
        if not os.path.lexists(dst):
            dst.symlink_to(SRC / f"{r['stem']}.png")
    print(f"{len(rows)} images, {len(subj)} subjects: train {len(train)} / val {len(val)} / test {len(test)}")
    for k, v in counts.items():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
