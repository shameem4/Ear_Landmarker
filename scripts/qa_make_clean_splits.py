"""Drop rows from the training splits without touching images or landmarks.

How to remove a row that is not an ear. NOT by deleting the image and NOT
by editing landmarks.npy: every QA artefact in this project -- gate.json,
remark.npz, earness.npz, outer_v2.npz, every rows.txt -- indexes rows by
position in the manifest, and removing a row would silently shift all of
them. The images are also the only copy.

Instead the split CSVs are rewritten without the dropped rows. The corpus
is unchanged and complete; training simply never sees them. Reverting is
pointing train.py back at the original CSV, and the drop list is a plain
text file under version control, so what was dropped and why stays
reviewable.

    python scripts/qa_make_clean_splits.py qa_lists/drop_rows.txt
    python train.py --data-dir data/manual --train-csv train_clean.csv

The drop list is CURATED, not thresholded. BlazeEar confidence ranks rows
by "is there an ear here", and it works -- row 7397, a striped shirt with
no ear in it, is 15th lowest of 10,535 -- but the bottom of its
distribution is mostly hard-and-valid: ears under red light, ears behind
white hair, high-contrast black and white, heavily degraded frames. Those
are exactly the rows the model needs, and dropping low-agreement rows has
already been measured in this project to remove hard IMAGES rather than
bad LABELS. So the detector proposes and a human disposes.
"""
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DROP = pathlib.Path(sys.argv[1])
DATA = ROOT / "data/manual"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "_clean"

drop = set()
for line in DROP.read_text().splitlines():
    line = line.split("#")[0].strip()
    if line:
        drop.update(int(x) for x in line.split())
print(f"{len(drop)} rows on the drop list")

for nm in ("train", "val", "test"):
    src = DATA / f"{nm}.csv"
    if not src.exists():
        continue
    rd = list(csv.DictReader(open(src)))
    keep = [r for r in rd if int(r.get("idx") or r.get("index")) not in drop]
    dst = DATA / f"{nm}{SUFFIX}.csv"
    with open(dst, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rd[0].keys())
        wr.writeheader(); wr.writerows(keep)
    print(f"  {nm}: {len(rd)} -> {len(keep)}  ({len(rd)-len(keep)} dropped)"
          f"  -> {dst.relative_to(ROOT)}")
