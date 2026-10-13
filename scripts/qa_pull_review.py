"""Turn the review artifact's stored verdicts into the drop list.

The decisions live in the artifact's own database, one document per row,
so they are READ BACK rather than retyped -- which was the point of
building the review page instead of a contact sheet. Save them with the
ArtifactData 'list' action and its out_dir, then point this at that
directory.

    python scripts/qa_pull_review.py <dir>/decisions
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = pathlib.Path(sys.argv[1])

by = {}
for f in D.glob("*.json"):
    # out_dir writes the document BODY, named <doc_id>.json -- the row id is
    # the filename, not a field inside it.
    v = (json.load(open(f)) or {}).get("verdict")
    if v:
        by.setdefault(v, []).append(int(f.stem))
for k in by:
    by[k].sort()
total = sum(len(v) for v in by.values())
print(f"{total} verdicts")
for k in sorted(by):
    print(f"  {k:<8}{len(by[k]):>4}")

z = np.load(ROOT / "runs/earness/earness.npz")
conf = {int(i): float(c) for i, c in zip(z["idx"], z["conf"])}
iou = {int(i): float(c) for i, c in zip(z["idx"], z["iou"])}

# The header is REWRITTEN, not appended to: re-running this must not stack
# another copy of it onto the file.
lines = [
    "# Rows dropped from training: no ear in the image.",
    "#",
    "# BlazeEar confidence ranks the candidates (runs/earness) and a human",
    "# decides -- the detector is a ranker, not a drop rule. Reviewed by hand",
    f"# in the Non-Ear Triage artifact, {total} verdicts, 2026-10-13; read back",
    "# from the artifact database rather than transcribed.",
    "#",
    "# Nothing here is deleted: scripts/qa_make_clean_splits.py rewrites the",
    "# split CSVs, leaving the images, the manifest and landmarks.npy intact,",
    "# so every QA artefact keeps its row indices.",
    "#",
]
for i in by.get("drop", []):
    lines.append(f"{i:<7}# conf {conf.get(i, float('nan')):.2f}  "
                 f"IoU {iou.get(i, float('nan')):.2f}")
(ROOT / "qa_lists/drop_rows.txt").write_text("\n".join(lines) + "\n")
print(f"\nwrote qa_lists/drop_rows.txt ({len(by.get('drop', []))} rows)")
for k in ("keep", "unsure"):
    if by.get(k):
        (ROOT / f"qa_lists/review_{k}.txt").write_text(" ".join(map(str, by[k])) + "\n")
        print(f"wrote qa_lists/review_{k}.txt ({len(by[k])})")
