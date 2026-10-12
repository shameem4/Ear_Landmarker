#!/usr/bin/env python3
"""Rebuild every derived 3D artefact in canonical millimetres, and retire the
mixed-unit originals.

The v4 files were built before the corpus had one unit system, so they carry
`ear_size` in each dataset's native units and landmarks in each fit's own
frame. Both have already caused wrong conclusions, so rather than leave them
where something can pick them up, the mm versions are written beside them and
the originals are MOVED to an archive directory.

MOVED, NOT DELETED, AND NOT OVERWRITTEN IN PLACE. Head 2 and head 4
checkpoints record their basis path in hparams, so silently changing the
CONTENTS of lm_basis_v4.npz would make every such checkpoint decode shape
coefficients against a differently scaled basis and report plausible nonsense.
Moving the file instead makes a stale reference fail loudly with a missing
path, which is the failure mode to prefer.

ccby_plus IS RE-TESTED, NOT JUST REBUILT. Its v4 note says "DO NOT USE AS A
SHAPE MODEL" because mode 1 separated HUTUBS from AudioEar3D at Cohen's
d = 2.01, attributed to AudioEar3D being "flat in depth". That diagnosis was
wrong -- the flatness was a frame mismatch, since AudioEar3D fits sat 23
degrees off the others and the depth span was being read along the wrong axis.
In one frame the spans agree within 3%. So the separation is measured again
here, and the verdict is whatever the number says.

    python scripts/regen_mm_artefacts.py
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import load_mesh                                # noqa: E402

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")
CAN = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/corpus/canonical_esc")
PY = sys.executable
RETIRE = ["lm3d_v4_all.npz", "lm3d_v4_hutubs.npz", "lm3d_v4_sonicom.npz",
          "lm3d_v4_audioear.npz", "ear_pca_v4_all.npz", "ear_pca_v4_ccby.npz",
          "ear_pca_v4_ccby_plus.npz", "lm_basis_v4.npz", "gt_coeffs_v4.npz",
          "ear_pca.npz", "ear_pca_all.npz", "ear_pca_ccby.npz",
          "mean_hutubs.obj", "mean_hutubs_smooth.obj",
          "corpus_landmarks_3d.npz", "mean_landmarks.npz"]


def run(args):
    r = subprocess.run([PY] + args, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(f"failed: {' '.join(args[:2])}")
    return r.stdout


def save_obj(path, V, F):
    with open(path, "w") as f:
        f.write("# canonical frame, millimetres\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def main():
    global CAN
    p = argparse.ArgumentParser()
    p.add_argument("--marks", default=str(TPL / "mean_v5_marks.npz"))
    p.add_argument("--template", default=str(TPL / "ear_mean_v4.obj"))
    p.add_argument("--archive", default=str(TPL / "archive_mixed_units"))
    p.add_argument("--canonical", default=str(CAN),
                   help="canonical corpus to rebuild from. Defaults to the "
                        "original; canonical_v7 is the adaptive re-fit "
                        "(fold-free hutubs+sonicom, audioear carried over "
                        "unchanged since it was never adaptively re-fit).")
    p.add_argument("--no-retire", action="store_true")
    a = p.parse_args()

    CAN = pathlib.Path(a.canonical)
    Vt, Ft = load_mesh(a.template)

    # ---- per-source mean ears, rebuilt in the canonical frame -------------
    print("per-source mean ears")
    for ds in ["sonicom", "hutubs", "audioear"]:
        V = np.mean([load_mesh(str(f))[0] for f in sorted((CAN / ds).glob("*.obj"))],
                    axis=0)
        save_obj(TPL / f"mean_{ds}_mm.obj", V, Ft)
        print(f"  mean_{ds}_mm.obj  extent {np.ptp(V, axis=0).round(1)} mm")
    V = np.mean([load_mesh(str(f))[0]
                 for ds in ["sonicom", "hutubs", "audioear"]
                 for f in sorted((CAN / ds).glob("*.obj"))], axis=0)
    save_obj(TPL / "mean_corpus_mm.obj", V, Ft)
    print(f"  mean_corpus_mm.obj extent {np.ptp(V, axis=0).round(1)} mm")

    # ---- landmarks, per source and pooled ---------------------------------
    print("\nlandmarks")
    sets = {"all": ["sonicom", "hutubs", "audioear"], "sonicom": ["sonicom"],
            "hutubs": ["hutubs"], "audioear": ["audioear"]}
    for tag, dss in sets.items():
        out = TPL / f"lm3d_v5_{tag}_mm.npz"
        run(["scripts/propagate_landmarks.py", "--marks", a.marks,
             "--template", a.template, "--fits"] + [str(CAN / d) for d in dss]
            + ["--out", str(out)])
        z = np.load(out, allow_pickle=True)
        print(f"  {out.name:28s} {z['landmarks'].shape}  ear "
              f"{z['ear_size'].min():.1f}-{z['ear_size'].max():.1f} mm")

    # ---- shape models, by licence -----------------------------------------
    print("\nshape models")
    pcas = {
        "all": (["sonicom", "hutubs", "audioear"],
                "canonical mm; HUTUBS + AudioEar3D (CC BY) + SONICOM (CC BY-SA)"
                " -- inherits CC BY-SA"),
        "ccby": (["hutubs"], "canonical mm; HUTUBS only, CC BY 4.0"),
        "ccby_plus": (["hutubs", "audioear"],
                      "canonical mm; HUTUBS + AudioEar3D, both CC BY 4.0"),
    }
    for tag, (dss, note) in pcas.items():
        out = TPL / f"ear_pca_v5_{tag}_mm.npz"
        run(["scripts/build_ear_pca.py", "--fits"] + [str(CAN / d) for d in dss]
            + ["--template", a.template, "--out", str(out), "--note", note])
        z = np.load(out, allow_pickle=True)
        print(f"  {out.name:28s} {z['n_ears']} ears, ear "
              f"{z['ear_size'].min():.1f}-{z['ear_size'].max():.1f} mm")

    # ---- is ccby_plus still modelling the scanner? ------------------------
    print("\nccby_plus re-test: does mode 1 still separate the two sources?")
    z = np.load(TPL / "ear_pca_v5_ccby_plus_mm.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    src = np.array(["hutubs" if n.startswith("pp") else "audioear"
                    for n in names])
    V = np.array([load_mesh(str(CAN / ("hutubs" if s == "hutubs" else "audioear")
                                / f"{n}.obj"))[0]
                  for n, s in zip(names, src)])
    from ear3d.wrap import rigid_fit
    A, m = V.copy(), V.mean(0)
    for _ in range(5):
        for i in range(len(A)):
            R, t, s = rigid_fit(A[i], m)
            A[i] = s * A[i] @ R.T + t
        m = A.mean(0)
    C = z["components"]
    coef = (A - m).reshape(len(A), -1) @ C.T
    frac = z["variance"] / z["variance"].sum()
    worst = []
    for k in range(8):
        h, e = coef[src == "hutubs", k], coef[src == "audioear", k]
        sp = np.sqrt(((len(h) - 1) * h.var(ddof=1)
                      + (len(e) - 1) * e.var(ddof=1)) / (len(h) + len(e) - 2))
        d = abs(h.mean() - e.mean()) / max(sp, 1e-12)
        worst.append((k, d, 100 * frac[k]))
        print(f"  mode {k + 1}: Cohen's d = {d:5.2f}   "
              f"({100 * frac[k]:4.1f}% of variance)")
    md = max(w[1] for w in worst)
    print(f"  -> largest separation d = {md:.2f}; v4 reported 2.01 on mode 1")
    print("     " + ("still source-driven, keep the warning" if md > 1.0 else
                     "the separation is GONE -- it was the frame mismatch, so"
                     "\n     ccby_plus is usable and the CC BY model doubles "
                     "to 224 ears"))

    # ---- landmark basis and ground-truth coefficients ---------------------
    print("\nlandmark basis and coefficients")
    run(["scripts/build_landmark_basis.py", "--pca",
         str(TPL / "ear_pca_v7_all_mm.npz"), "--marks", a.marks,
         "--template", a.template, "--out", str(TPL / "lm_basis_v5_95_unitheight.npz")])
    zb = np.load(TPL / "lm_basis_v5_95_unitheight.npz", allow_pickle=True)
    print(f"  lm_basis_v5_95_unitheight.npz  {zb['components'].shape}, "
          f"ear_size {float(zb['ear_size']):.1f} mm")

    # ---- retire the mixed-unit originals ----------------------------------
    if not a.no_retire:
        arch = pathlib.Path(a.archive)
        arch.mkdir(parents=True, exist_ok=True)
        moved = []
        for f in RETIRE:
            s = TPL / f
            if s.exists():
                shutil.move(str(s), str(arch / f))
                moved.append(f)
        (arch / "README.md").write_text(
            "# Retired: mixed-unit artefacts\n\n"
            "Built before the corpus had one unit system. `ear_size` is in "
            "each dataset's native units (SONICOM mm, HUTUBS per-head "
            "normalised) and landmarks are in each fit's own frame, so "
            "cross-source comparisons made with these files are wrong -- that "
            "is how AudioEar3D came to be called 'flat in depth'.\n\n"
            "They are MOVED rather than deleted or overwritten because head 2 "
            "and head 4 checkpoints record their basis path in hparams. "
            "Rewriting `lm_basis_v4.npz` in place would have made those "
            "checkpoints decode against a differently scaled basis and report "
            "plausible nonsense; a missing path fails loudly instead.\n\n"
            "Use the `*_v5_*_mm` files. To interpret an old checkpoint, point "
            "it at this directory deliberately.\n\n"
            + "".join(f"- {f}\n" for f in moved))
        print(f"\nretired {len(moved)} files to {arch}")
        for f in moved:
            print(f"  {f}")


if __name__ == "__main__":
    main()
