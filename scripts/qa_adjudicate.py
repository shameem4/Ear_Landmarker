"""Use the coarse independent instrument as a DISCRIMINATOR, not a generator.

Gemini's shape fit has a ~5-6% of ear height noise floor, three to five
times coarser than the landmarker's agreement with the annotations, so it
cannot say where a landmark belongs ([[flaggers-not-correctors]]). That
killed the idea of correcting rows with it.

But choosing between two CANDIDATES is a much easier question than
producing coordinates. Where a row is disputed we have two: the
annotation, and the model's own prediction. They differ by more than 15%
of ear height -- three times the instrument's floor -- so a box that is
far too coarse to generate an answer is still sharp enough to say which
of the two it contains.

That is what this measures. For every disputed row, the excursion of the
annotation outside the Gemini box, against the excursion of the model's
prediction outside the same box.

What a win here does and does NOT license. Inside-the-box is NECESSARY,
not sufficient: a prediction can sit well inside a box and still be
wrong, because a box constrains only extent. So this can establish that
the annotation is grossly wrong and the prediction is not wrong in the
same way. It cannot establish that the prediction is right.

Split train vs held-out and report apart, as always: on a training row
the prediction is partly memorised. Note these rows are the ones the
model FAILED to memorise, which is the signal, but the caveat stands.

    python scripts/qa_adjudicate.py runs/adjudicate --tol 15
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from inference import LandmarkPredictor                      # noqa: E402
from qa_make_drop_splits import excursion                    # noqa: E402

# Pinned, never find_best_checkpoint(): the drop-experiment runs are in
# runs/checkpoints too and a half-trained arm would quietly win the scan.
SHIPPED = "runs/checkpoints/manual_occ_s42/EarLandmarker_292_nme=0.0149.ckpt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tol", type=float, default=15.0)
    p.add_argument("--ckpt", default=SHIPPED)
    a = p.parse_args()
    OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lmk = LandmarkPredictor(ROOT / a.ckpt, device=dev)
    print(f"landmarker: {a.ckpt}")

    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    split = {}
    for nm in ("train", "val", "test"):
        for r in csv.DictReader(open(ROOT / "data/manual" / f"{nm}.csv")):
            split[int(r["idx"])] = nm

    res = {}
    for arm in ("flagged", "control"):
        B = json.load(open(ROOT / f"runs/measure/{arm}/boxes_gemini_nothink.json"))
        for k, box in B.items():
            i = int(k)
            g = L[i]
            if not np.isfinite(g).all():
                continue
            e_lab = excursion(box, g)
            if not np.isfinite(e_lab) or e_lab <= a.tol:
                continue                      # not disputed; nothing to adjudicate
            try:
                im = np.asarray(Image.open(
                    ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB"))
            except Exception:
                continue
            h, w = im.shape[:2]
            pr = lmk.predict(im) / np.array([w, h])        # -> normalised
            e_prd = excursion(box, pr)
            res[k] = dict(arm=arm, split=split.get(i, "?"),
                          e_label=e_lab, e_pred=e_prd,
                          pred=pr.tolist(), box=box)
    json.dump(res, open(OUT / "adjudicate.json", "w"))

    print(f"\n{len(res)} disputed rows adjudicated "
          f"(annotation > {a.tol:.0f}% outside the box)\n")
    print(f"{'slice':<22}{'n':>5}{'label':>9}{'pred':>9}{'pred wins':>11}"
          f"{'pred inside':>13}")
    def line(nm, sel):
        v = [r for r in res.values() if sel(r)]
        if not v:
            return
        el = np.array([r["e_label"] for r in v])
        ep = np.array([r["e_pred"] for r in v])
        print(f"{nm:<22}{len(v):>5}{np.median(el):>9.1f}{np.median(ep):>9.1f}"
              f"{100*(ep < el).mean():>10.0f}%{100*(ep <= a.tol).mean():>12.0f}%")
    line("all", lambda r: True)
    line("  train", lambda r: r["split"] == "train")
    line("  held-out", lambda r: r["split"] in ("val", "test"))
    line("  SAM-flagged", lambda r: r["arm"] == "flagged")
    line("  SAM missed", lambda r: r["arm"] == "control")
    print("\n(excursion outside the Gemini box, % of ear height."
          "\n 'pred inside' = the model's prediction is no longer disputed,"
          "\n which is NECESSARY but not sufficient for it to be correct.)")


if __name__ == "__main__":
    main()
