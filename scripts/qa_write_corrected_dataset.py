"""Write the silhouette-corrected landmark array as a SEPARATE dataset.

Separate, but not a separate directory: train.py already takes
--landmarks, so the corrected array lives beside the original as

    data/manual/landmarks_sil_corrected.npy

and is trained with

    python train.py --data-dir data/manual --landmarks landmarks_sil_corrected.npy

The images, the manifest and the three split CSVs are byte-identical to
the shipped ones, so the only difference between a corrected run and a
baseline run is the 20 outer points on gated rows. Duplicating the image
tree would have bought nothing and cost 10,535 files.

What changes: points 0-19 on the rows corrected by
qa_correct_outer_pass.py. Points 20-54 are UNTOUCHED -- the interior
propagation is a separate question (see qa_outer_inner_clash.py), and
rows outside the gate keep their original annotation because the
silhouette is not entitled to judge them.

    python scripts/qa_write_corrected_dataset.py runs/corrected
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN = pathlib.Path(sys.argv[1])
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "landmarks_sil_corrected.npy"

L = np.load(ROOT / "data/manual/landmarks.npy")
z = np.load(RUN / "outer.npz")
idx, pts = z["idx"].astype(int), z["pts"]

C = L.copy()
d = np.abs(C[idx, :20] - pts)
C[idx, :20] = pts
np.save(ROOT / "data/manual" / OUTNAME, C)

moved = (np.linalg.norm(L[idx, :20] - pts, axis=2) > 1e-9)
print(f"wrote data/manual/{OUTNAME}")
print(f"  rows changed      {len(idx)} of {len(L)}")
print(f"  points changed    {moved.sum()} of {moved.size} on those rows "
      f"({100*moved.mean():.1f}%)")
print(f"  rows with 0 moved {int((~moved.any(1)).sum())}")
assert np.allclose(C[:, 20:], L[:, 20:]), "interior must be untouched"
unchanged = np.setdiff1d(np.arange(len(L)), idx)
assert np.allclose(C[unchanged], L[unchanged]), "non-gated rows must be untouched"
print("  interior and non-gated rows verified identical")
(RUN / "corrected_rows.txt").write_text(" ".join(map(str, sorted(idx))))
