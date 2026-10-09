#!/usr/bin/env python3
"""Split the rendered depth corpus BY SUBJECT, not by image.

Two reasons it cannot be by image. Every ear appears at 21 poses, so a random
split puts near-identical views of the same ear on both sides and the test
score measures memorisation. And a person's two ears are closely alike -- this
project measured left against right agreeing to 1.3-1.5% of ear size, far
closer than two people -- so the sides must travel together too.

TWO WAYS TO HANDLE SOURCE, and the difference matters enough to keep both.

  --balance off (default): stratified so each split holds the same PROPORTION
  of each source as the corpus. The corpus is 76% SONICOM, so each split is
  too, and a pooled metric is three-quarters a statement about SONICOM.

  --balance on: val and test hold the same NUMBER of subjects from each
  source, sized by the smaller source. The pooled number then weights the two
  sources equally, and each per-source number carries the same error bar, so
  they can be compared to each other rather than only to themselves.

Balancing the split does NOT remove the source shortcut. A logistic regression
on 40 shape coefficients names the source of a held-out subject 93.9% of the
time, and nothing about how the rows are divided changes what is in a render.
What balancing buys is that the shortcut stops being rewarded by the prior and
starts being visible in the metric. The number that actually measures the cost
is leave-one-source-out: train on one source, test on the other.

--holdout EXISTS SO THOSE THREE RECIPES CAN BE COMPARED AT ALL. Each one holds
out different subjects, so each one's own test score is a statement about a
different set of ears, and the differences between them are then part model
and part test set in unknown proportion. The first version of this experiment
was built without it and had to be thrown away: of the three test sets, no two
shared a single subject that all three recipes had kept out of training.

With --holdout N, the SAME N subjects per source are removed first, with their
own seed, before anything else happens -- before --only-source, before
balancing -- and written to <prefix>common_test.csv. Every recipe then trains
on a subset of the remainder and is scored on identical ears, so a difference
between two runs is the recipe. The holdout seed is separate from --seed on
purpose: changing how rows are split must not change which ears are the
yardstick.

    python scripts/split_depth3d.py --data data/depth3d_v5 --holdout 14
    python scripts/split_depth3d.py --data data/depth3d_v5 --holdout 14 \
        --balance --prefix bal_
"""
import argparse
import collections
import csv
import io
import pathlib
import sys

import numpy as np


def subject_of(ear):
    """pp10_left -> pp10;  P0001_right -> P0001."""
    return ear.rsplit("_", 1)[0]


def source_of(subject):
    """HUTUBS subjects are ppNN; SONICOM are PNNNN."""
    return "hutubs" if subject.startswith("pp") else "sonicom"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--balance", action="store_true",
                   help="equal SUBJECT COUNT per source in val and test, "
                        "sized by the smaller source")
    p.add_argument("--balance-frac", type=float, default=0.25,
                   help="with --balance, the fraction OF THE SMALLER SOURCE "
                        "that goes to each of val and test")
    p.add_argument("--only-source", default="",
                   help="keep just this source, for leave-one-source-out")
    p.add_argument("--holdout", type=int, default=0,
                   help="remove this many subjects PER SOURCE first and write "
                        "them to <prefix>common_test.csv, so several split "
                        "recipes can be scored on identical ears")
    p.add_argument("--holdout-seed", type=int, default=1234,
                   help="deliberately separate from --seed: which ears are "
                        "the yardstick must not move when the split moves")
    p.add_argument("--prefix", default="", help="written as <prefix>train.csv")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / "manifest.csv")))
    if not rows:
        sys.exit("empty manifest")
    source = {}
    for r in rows:
        s = subject_of(r["ear"])
        source[s] = source_of(s)

    held = set()
    if a.holdout:
        # FIRST, and on its own seed, so every recipe excludes the same ears.
        pool = collections.defaultdict(list)
        for s, v in source.items():
            pool[v].append(s)
        hrng = np.random.default_rng(a.holdout_seed)
        for ds in sorted(pool):
            subs = sorted(pool[ds])
            if a.holdout > len(subs):
                sys.exit(f"--holdout {a.holdout} exceeds the {len(subs)} "
                         f"{ds} subjects available")
            hrng.shuffle(subs)
            held.update(subs[:a.holdout])
        hrows = [r for r in rows if subject_of(r["ear"]) in held]
        fields_h = list(rows[0].keys())
        if "source" not in fields_h:
            fields_h.append("source")
        # ONE file, unprefixed, because by construction every recipe computes
        # the same holdout. If a later invocation would write something
        # different, the yardstick has moved and that must not pass quietly.
        hp = d / "common_test.csv"
        body = io.StringIO()
        w = csv.DictWriter(body, fieldnames=fields_h, lineterminator="\n")
        w.writeheader()
        w.writerows({**r, "source": source[subject_of(r["ear"])]} for r in hrows)
        text = body.getvalue()
        if hp.exists() and hp.read_text() != text:
            sys.exit(f"{hp} already exists and differs from the holdout this "
                     f"run computed. Two recipes would be scored on different "
                     f"ears, which is the one thing --holdout exists to "
                     f"prevent. Delete it deliberately, or match --holdout "
                     f"({a.holdout}) and --holdout-seed ({a.holdout_seed}).")
        hp.write_text(text)
        print(f"holdout: {len(held)} subjects ({a.holdout} per source), "
              f"{len(hrows)} views -> common_test.csv")
        rows = [r for r in rows if subject_of(r["ear"]) not in held]
        source = {s: v for s, v in source.items() if s not in held}

    if a.only_source:
        if a.only_source not in set(source.values()):
            sys.exit(f"--only-source {a.only_source!r} is not one of "
                     f"{sorted(set(source.values()))}")
        rows = [r for r in rows if source[subject_of(r["ear"])] == a.only_source]
        source = {s: v for s, v in source.items() if v == a.only_source}

    by_source = collections.defaultdict(list)
    for s, v in source.items():
        by_source[v].append(s)

    if a.balance:
        n_eval = int(round(a.balance_frac * min(len(v) for v in by_source.values())))
        if n_eval < 2:
            sys.exit(f"--balance-frac {a.balance_frac} leaves {n_eval} subjects "
                     f"per source per split; too few to measure anything")
        print(f"balanced: {n_eval} subjects per source in each of val and test")

    rng = np.random.default_rng(a.seed)
    assign = {}
    for ds in sorted(by_source):
        subs = sorted(by_source[ds])
        rng.shuffle(subs)
        n = len(subs)
        if a.balance:
            nv = nt = n_eval
        else:
            nv, nt = int(round(a.val * n)), int(round(a.test * n))
        for s in subs[:nt]:
            assign[s] = "test"
        for s in subs[nt:nt + nv]:
            assign[s] = "val"
        for s in subs[nt + nv:]:
            assign[s] = "train"

    # `source` is written into every row so a metric can be grouped by it
    # without re-deriving the rule from the subject name in a second place.
    fields = list(rows[0].keys())
    if "source" not in fields:
        fields.append("source")
    out = {k: [] for k in ("train", "val", "test")}
    for r in rows:
        s = subject_of(r["ear"])
        out[assign[s]].append({**r, "source": source[s]})
    for k, v in out.items():
        with open(d / f"{a.prefix}{k}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(v)

    print(f"{len(rows)} views, {len(assign)} subjects "
          f"-> {d}/{a.prefix}{{train,val,test}}.csv")
    srcs = sorted(by_source)
    for k in ("train", "val", "test"):
        subs = [s for s in assign if assign[s] == k]
        ears = {r["ear"] for r in out[k]}
        per = "  ".join(
            f"{ds} {sum(source[s] == ds for s in subs):3d}s/"
            f"{sum(r['source'] == ds for r in out[k]):5d}v" for ds in srcs)
        print(f"  {k:<6s} {len(out[k]):6d} views  {len(ears):4d} ears  "
              f"{len(subs):4d} subjects   {per}")

    # the check that matters: no subject may appear in two splits
    seen = collections.defaultdict(set)
    for k in out:
        for r in out[k]:
            seen[subject_of(r["ear"])].add(k)
    leaked = {s: v for s, v in seen.items() if len(v) > 1}
    print(f"subjects appearing in more than one split: {len(leaked)}"
          + ("  <-- LEAK" if leaked else "  (none, as required)"))
    if leaked:
        sys.exit(1)


if __name__ == "__main__":
    main()
