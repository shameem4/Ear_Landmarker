#!/usr/bin/env python3
"""Put every ear in millimetres, in one frame, reversibly.

THREE unit systems were in play and the mixture caused four separate errors in
a single session -- a PCA scale off by 100x, a landmark depth span that read a
third of the others and was called "flat", and twice a figure quoted against
the wrong denominator:

  SONICOM fits   millimetres, from the raw STL.
  AudioEar3D     millimetres.
  HUTUBS fits    PER-HEAD NORMALISED units. ear3d.frames.load_head divides by
                 max|V - mean| of that subject's head, so the factor differs
                 for EVERY SUBJECT and is not recorded in the cut transform,
                 whose translation is in those same normalised units.
                 Recovering millimetres means re-reading the original .ply.
                 Without it a HUTUBS ear reads as larger than its own head.

CANONICAL FRAME is the template's. Every fit shares its topology, so the
alignment is Procrustes over corresponding vertices and needs no search. The
transform is ROTATION AND TRANSLATION ONLY: scale is deliberately not removed,
because keeping true millimetres is the point. rigid_fit's reflection guard
stays on, so a right ear is aligned to a MIRRORED template rather than being
quietly turned into a left one. Which mirror fits better is measured per ear,
so a mislabelled file cannot land in the wrong frame.

EVERYTHING IS REVERSIBLE, AND THE REVERSAL IS CHECKED. Each ear gets a 4x4
taking its canonical mesh back to original coordinates, unit scale included.
An ear that does not return to within --tol is reported and not written.

THE UNITS ARE CHECKED AGAINST DATA THIS PIPELINE NEVER SAW. HUTUBS ships
caliper anthropometry, so recovered pinna height is compared with the measured
d5. Plausibility is what let the mixed units survive this long; an external
measurement is what actually fails when a per-subject factor is wrong.

THE CHECK MEASURES BETWEEN LANDMARKS, NOT ACROSS THE BOUNDING BOX. d5 is a
caliper height from the top of the helix to the bottom of the lobule, and
`ear_mm` -- max extent of the whole fitted template -- is not that: the
template carries geometry past the pinna, so the box overshoots d5 by 4%
even on a good corpus, and by 11% on no-control-point fits, whose surplus
floats further out. That is a difference in what is measured, not in units:
the same fits give 1.012 +- 0.029 of d5 landmark-to-landmark. `ear_mm` is
still reported, because it is the number every earlier manifest carries,
but `pinna_mm` is what the check gates on.

    python scripts/canonicalise_corpus.py
"""
import argparse
import csv
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.wrap import base_topology, load_mesh, rigid_fit     # noqa: E402
from ear3d.scheme import GROUPS, GROUP_NAMES                   # noqa: E402
from gt_landmark_check import DEFAULT_MARKS, read_marks        # noqa: E402

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")
CORPUS = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/corpus")
ANTH = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/"
                    "hutubs_extras/AntrhopometricMeasures.csv")
SOURCES = {"sonicom": "faceform2_sonicom", "hutubs": "faceform2_hutubs",
           "audioear": "ae3d_fits"}


def pinna_height(V, F, tri_m, bary_m):
    """Top of the helix to the bottom of the lobule, along the long axis.

    The caliper definition of HUTUBS d5. Taken along the canonical long
    axis rather than straight-line because that is how a caliper held
    against a head measures it, and it tracks d5 slightly better
    (1.012 vs 1.031 of measured, over the 112 HUTUBS ears).
    """
    g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
    helix = g["ascending_helix"] + g["helix"] + g["descending_helix"]
    L = np.einsum("ij,ijk->ik", bary_m, V[F[tri_m]])
    ax = int(np.argmax(np.ptp(V, axis=0)))
    return float(abs(L[helix][:, ax].max() - L[g["ear_lobe"]][:, ax].min()))


def to_mm_scale(dataset, name, tf_dir, cache, assume_mm=False):
    """Factor taking this fit's stored coordinates to millimetres."""
    if assume_mm:
        return 1.0, "already mm"
    if dataset != "hutubs":
        return 1.0, ""
    f = tf_dir / f"{name}.npz"
    if not f.exists():
        return None, f"no cut transform at {f.name}"
    src = str(np.load(f, allow_pickle=True)["source"])
    if src in cache:
        return cache[src], src
    import open3d as o3d
    H = np.asarray(o3d.io.read_triangle_mesh(src).vertices)
    if len(H) == 0:
        return None, f"could not read {src}"
    k = float(np.abs(H - H.mean(0)).max()) * 1000.0   # load_head divisor, m->mm
    cache[src] = k
    return k, src


def save_obj(path, V, F):
    with open(path, "w") as f:
        f.write("# canonical frame, millimetres\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=str(TPL / "ear_mean_v4.obj"))
    p.add_argument("--out-root", default=str(CORPUS / "canonical"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--assume-mm", action="store_true",
                   help="inputs are already in millimetres, e.g. the fits "
                        "from wrap_batch_head.py; skip the per-subject "
                        "HUTUBS recovery")
    p.add_argument("--sources", nargs="*", default=None,
                   help="dataset=dir pairs, overriding the defaults")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="max round-trip error in mm before an ear is rejected")
    a = p.parse_args()

    sources = (dict(s.split("=", 1) for s in a.sources) if a.sources
               else SOURCES)
    Vt, Ft = load_mesh(a.template)
    tri_m, bary_m, _ = read_marks(DEFAULT_MARKS)
    tri_m, bary_m = tri_m[:55], bary_m[:55]
    mirrors = {"z": np.diag([1.0, 1.0, -1.0]), "x": np.diag([-1.0, 1.0, 1.0])}
    out_root = pathlib.Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows, bad, cache = [], [], {}

    for dataset, sub in sources.items():
        d = pathlib.Path(sub) if '/' in sub else CORPUS / sub
        tf_dir = CORPUS / f"tf_{dataset}"
        files = sorted(d.glob("*.obj"))
        if a.limit:
            files = files[:a.limit]
        od = out_root / dataset
        od.mkdir(parents=True, exist_ok=True)
        n0 = len(rows)
        for f in files:
            name = f.stem
            V, F = load_mesh(str(f))
            try:
                V, F = base_topology(V, F, Vt, Ft, name)
            except ValueError as e:
                bad.append((name, str(e)))
                continue
            k, note = to_mm_scale(dataset, name, tf_dir, cache,
                                  a.assume_mm)
            if k is None:
                bad.append((name, note))
                continue
            Vmm = V * k

            best = None
            for tag, M in [("none", np.eye(3))] + list(mirrors.items()):
                tgt = Vt @ M
                R, t, _ = rigid_fit(Vmm, tgt, with_scale=False)
                res = float(np.linalg.norm(Vmm @ R.T + t - tgt, axis=1).mean())
                if best is None or res < best[0]:
                    best = (res, tag, R, t)
            res, tag, R, t = best
            Vc = Vmm @ R.T + t

            back = np.eye(4)                      # canonical -> original
            back[:3, :3] = R.T / k
            back[:3, 3] = -(t @ R) / k
            err = float(np.abs(Vc @ back[:3, :3].T + back[:3, 3] - V).max())
            if err > a.tol:
                bad.append((name, f"round trip {err:.2e}"))
                continue

            save_obj(od / f"{name}.obj", Vc, F)
            np.savez(od / f"{name}.npz", to_original=back, to_mm=k,
                     rotation=R, translation=t, mirror=tag, residual_mm=res,
                     roundtrip=err, dataset=dataset, source=str(f))
            rows.append({"name": name, "dataset": dataset, "to_mm": k,
                         "mirror": tag, "side": name.rsplit("_", 1)[-1],
                         "ear_mm": float(np.ptp(Vc, axis=0).max()),
                         "pinna_mm": pinna_height(Vc, F, tri_m, bary_m),
                         "align_resid_mm": res, "roundtrip": err})
        print(f"{dataset:9s} {len(rows) - n0:4d} ears -> {od}")

    # MERGE, DO NOT REWRITE. --sources names a subset, and a plain rewrite
    # drops the provenance rows of every dataset not in this run -- the to_mm
    # factor, the mirror and the round-trip check, which is the only record
    # that an ear's units were recovered correctly. Rows for the datasets
    # rebuilt here replace their old ones; the rest are carried over.
    man = out_root / "manifest.csv"
    if man.exists():
        done = {r["dataset"] for r in rows}
        prev = [r for r in csv.DictReader(open(man))
                if r.get("dataset") not in done]
        if prev:
            print(f"carried {len(prev)} manifest rows for "
                  f"{sorted({r['dataset'] for r in prev})}")
            # CARRIED ROWS COME BACK FROM CSV AS STRINGS. Every numeric
            # column has to be coerced here, or a run that recomputes only
            # one dataset mixes str and float and the summaries below die
            # with a ufunc type error -- which is exactly how the first
            # audioear merge failed, twice, one statistic at a time.
            num = {"to_mm", "ear_mm", "pinna_mm", "align_resid_mm", "roundtrip"}
            def _coerce(r):
                out = {}
                for k in rows[0]:
                    v = r.get(k, "")
                    if k in num and v != "":
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                    out[k] = v
                return out
            rows = rows + [_coerce(r) for r in prev]
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} canonicalised, {len(bad)} rejected")
    for n, why in bad[:8]:
        print(f"  ! {n}: {why}")
    import collections
    print("\nmirror chosen, against the filename's side:")
    for (side, mir), n in sorted(collections.Counter(
            (r["side"], r["mirror"]) for r in rows).items()):
        print(f"  {side:6s} -> {mir:5s} {n:4d}")
    for ds in sources:
        e = np.array([r["ear_mm"] for r in rows if r["dataset"] == ds])
        q = np.array([r["align_resid_mm"] for r in rows if r["dataset"] == ds])
        if len(e):
            print(f"\n{ds:9s} n={len(e):3d}  ear {e.mean():5.1f} mm "
                  f"[{e.min():.1f}, {e.max():.1f}]   "
                  f"alignment residual {q.mean():.2f} mm")

    if ANTH.exists():
        rec = {int(r["SubjectID"]): r for r in csv.DictReader(open(ANTH))}
        mine, ref = [], []
        for r in rows:
            m = re.match(r"pp(\d+)_(left|right)$", r["name"])
            if r["dataset"] != "hutubs" or not m:
                continue
            sid, side = int(m.group(1)), m.group(2)
            if sid in rec:
                mine.append(float(r["pinna_mm"]))
                ref.append(10.0 * float(
                    rec[sid][("L_" if side == "left" else "R_") + "d5"]))
        if mine:
            mine, ref = np.array(mine), np.array(ref)
            ratio = mine / ref
            print(f"\nUNIT CHECK against HUTUBS caliper anthropometry "
                  f"(d5, pinna height), n={len(mine)}")
            print("  measured helix-top to lobule-bottom, not bounding box")
            print(f"  recovered {mine.mean():.1f} mm vs measured {ref.mean():.1f} mm")
            print(f"  ratio {ratio.mean():.3f} +- {ratio.std():.3f}   "
                  f"r = {np.corrcoef(mine, ref)[0, 1]:+.3f}   "
                  f"mean abs error {np.abs(mine - ref).mean():.2f} mm")
            if abs(ratio.mean() - 1) > 0.10:
                sys.exit("  ! over 10% off -- the unit recovery is WRONG")

    rt = np.array([r["roundtrip"] for r in rows])
    print(f"\nworst round trip back to original coordinates: {rt.max():.2e}")
    print(f"wrote {out_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
