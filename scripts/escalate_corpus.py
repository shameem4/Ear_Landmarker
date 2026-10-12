#!/usr/bin/env python3
"""Pick, per ear, between the no-control-point fit and the ladder's rescue.

No-control-point Faceform is the best single pass over the corpus -- 476
ears fold-free at 80.7% once the canal ring is masked -- but a fifth of
the ears still fold, and the parameter ladder clears 77 of those 92. It
also costs accuracy, so it is not applied everywhere: this takes the
ladder's fit ONLY where it has STRICTLY fewer folds, ties going to the
no-CP arm, which makes escalation unable to make any ear worse.

FOLDS ARE COUNTED WITH CANAL RING 1 MASKED, at base resolution. The
template's canal is a protruding tube that no scan has, so it is crushed
flat by every fit and folds on over half the corpus; masking it after the
fit costs no coverage, because there is nothing there to cover.

THE 90-DEGREE TEST IS REPORTED BOTH WAYS. An absolute dihedral past 90 is
what the pipeline has always counted, but the template is not flat: its
sharpest crease rests at 52 degrees, so that test fires on a 38-degree
bend and calls the template's own anatomy a defect. `rel` is the honest
number -- the bend the FIT added on top of the template's rest angle.

    python scripts/escalate_corpus.py --nocp corpus/nocp_v1 \
        --ladder corpus/ladder92 --out corpus/escalated_v1
"""
import argparse
import collections
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WRAP = ROOT.parent / "wrap_work"
from ear3d.wrap import load_mesh                                   # noqa: E402


def face_pairs(F):
    adj = collections.defaultdict(list)
    for t, f in enumerate(F):
        for i in range(3):
            adj[tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))].append(t)
    return np.array([ts for ts in adj.values() if len(ts) == 2])


def dihedral(V, F, pairs):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", n[pairs[:, 0]], n[pairs[:, 1]]), -1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nocp", required=True,
                   help="dir of no-control-point fits, or a parent of "
                        "<dataset>/ subdirs")
    p.add_argument("--ladder", required=True, help="dir of ladder re-fits")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default=str(WRAP / "templates/ear_L_template.obj"))
    p.add_argument("--mask", default=str(WRAP / "templates/canal_mask_ring1.npz"))
    p.add_argument("--datasets", nargs="*", default=["hutubs", "sonicom"])
    a = p.parse_args()

    Vt, Ft = load_mesh(a.template)
    nb = len(Vt)
    keep = np.load(a.mask, allow_pickle=True)["keep_faces"]
    Fk = Ft[keep]
    PK = face_pairs(Fk)
    REST = dihedral(Vt, Fk, PK)

    out = pathlib.Path(a.out)
    nocp, lad = pathlib.Path(a.nocp), pathlib.Path(a.ladder)
    manifest, dirty, reverted = [], [], []
    for ds in a.datasets:
        src = nocp / ds if (nocp / ds).is_dir() else nocp
        (out / ds).mkdir(parents=True, exist_ok=True)
        for q in sorted(src.glob("*.obj")):
            name = q.stem
            Va = load_mesh(q)[0][:nb]
            fa = dihedral(Va, Fk, PK)
            best, arm, ang = q, "nocp", fa
            lp = lad / f"{name}.obj"
            if lp.exists():
                fb = dihedral(load_mesh(lp)[0][:nb], Fk, PK)
                if int((fb > 90).sum()) < int((fa > 90).sum()):
                    best, arm, ang = lp, "ladder", fb
                else:
                    reverted.append((name, int((fa > 90).sum()),
                                     int((fb > 90).sum())))
            shutil.copyfile(best, out / ds / f"{name}.obj")
            n_abs, n_rel = int((ang > 90).sum()), int(((ang - REST) > 90).sum())
            manifest.append(dict(ear=name, ds=ds, arm=arm, folds=n_abs,
                                 folds_rel=n_rel, maxdih=round(float(ang.max()), 1),
                                 escalated=int(lp.exists())))
            if n_abs:
                dirty.append((name, ds, arm, n_abs, n_rel, float(ang.max())))

    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    dirty.sort(key=lambda r: -r[4])
    with open(out / "dirty.txt", "w") as fh:
        fh.write("# ear\tdataset\tarm\tfolds_abs\tfolds_rel\tmax_dihedral\n")
        for r in dirty:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]:.0f}\n")

    n = len(manifest)
    nl = sum(1 for r in manifest if r["arm"] == "ladder")
    ab = sum(r["folds"] for r in manifest)
    rl = sum(r["folds_rel"] for r in manifest)
    print(f"{n} ears -> {out}")
    print(f"  arm: ladder {nl}  no-CP {n - nl} "
          f"({len(reverted)} escalated then reverted)")
    print(f"  absolute gate: {ab} folds over "
          f"{sum(1 for r in manifest if r['folds'])} ears "
          f"({sum(1 for r in manifest if not r['folds']) / n * 100:.1f}% clean)")
    print(f"  relative gate: {rl} folds over "
          f"{sum(1 for r in manifest if r['folds_rel'])} ears "
          f"({sum(1 for r in manifest if not r['folds_rel']) / n * 100:.1f}% clean)")
    print(f"  dirty list -> {out / 'dirty.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
