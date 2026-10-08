#!/usr/bin/env python3
"""Split the rendered depth corpus BY SUBJECT, not by image.

Two reasons it cannot be by image. Every ear appears at 21 poses, so a random
split puts near-identical views of the same ear on both sides and the test
score measures memorisation. And a person's two ears are closely alike -- this
project measured left against right agreeing to 1.3-1.5% of ear size, far
closer than two people -- so the sides must travel together too.

Stratified by dataset, because HUTUBS and SONICOM differ in mesh resolution and
in how the ear was cut, and an unlucky draw would confound the comparison.

    python scripts/split_depth3d.py --data data/depth3d
"""
import argparse
import collections
import csv
import pathlib
import sys

import numpy as np


def subject_of(ear):
    """pp10_left -> pp10;  P0001_right -> P0001."""
    return ear.rsplit("_", 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / "manifest.csv")))
    if not rows:
        sys.exit("empty manifest")
    dataset = {}
    for r in rows:
        s = subject_of(r["ear"])
        dataset[s] = "hutubs" if s.startswith("pp") else "sonicom"

    rng = np.random.default_rng(a.seed)
    assign = {}
    for ds in sorted(set(dataset.values())):
        subs = sorted(s for s in dataset if dataset[s] == ds)
        rng.shuffle(subs)
        n = len(subs)
        nv, nt = int(round(a.val * n)), int(round(a.test * n))
        for s in subs[:nt]:
            assign[s] = "test"
        for s in subs[nt:nt + nv]:
            assign[s] = "val"
        for s in subs[nt + nv:]:
            assign[s] = "train"

    counts = collections.Counter()
    out = {k: [] for k in ("train", "val", "test")}
    for r in rows:
        k = assign[subject_of(r["ear"])]
        out[k].append(r)
        counts[k] += 1
    for k, v in out.items():
        with open(d / f"{k}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(v)

    print(f"{len(rows)} views, {len(assign)} subjects")
    for k in ("train", "val", "test"):
        subs = [s for s in assign if assign[s] == k]
        ears = {r["ear"] for r in out[k]}
        print(f"  {k:<6s} {counts[k]:6d} views  {len(ears):4d} ears  "
              f"{len(subs):4d} subjects  "
              f"({sum(dataset[s] == 'hutubs' for s in subs)} hutubs, "
              f"{sum(dataset[s] == 'sonicom' for s in subs)} sonicom)")
    # the check that matters: no subject may appear in two splits
    seen = collections.defaultdict(set)
    for k in out:
        for r in out[k]:
            seen[subject_of(r["ear"])].add(k)
    leaked = {s: v for s, v in seen.items() if len(v) > 1}
    print(f"subjects appearing in more than one split: {len(leaked)}"
          + ("  <-- LEAK" if leaked else "  (none, as required)"))


if __name__ == "__main__":
    main()
