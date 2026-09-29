"""Score saved checkpoints on the held-out test split.

Exists so the numbers in README.md are reproducible rather than transcribed from
a console. Several early v2 runs predate the `trainer.test()` call in train.py
and so never logged `test/nme`; this fills them in from their checkpoints.

Uses the LightningModule's own test_step, so the numbers are identical in
definition to the ones the later runs logged themselves -- NME is mean L2 in
normalised [0,1] coordinates over visible points, not a re-derivation.

Usage:
    python scripts/eval_test.py                       # every run with a checkpoint
    python scripts/eval_test.py v6_persp65 v2_heatmap
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import EarLandmarkDataset          # noqa: E402
from model.lightning_module import EarLandmarkerModule  # noqa: E402

DATA_DIR = ROOT / "data" / "preprocessed"
CKPT_DIR = ROOT / "runs" / "checkpoints"


def best_ckpt(run: str) -> Path | None:
    """Lowest-NME checkpoint for a run, by the nme= field in its filename."""
    cks = [c for c in (CKPT_DIR / run).glob("*.ckpt") if "nme=" in c.name]
    if not cks:
        return None
    return min(cks, key=lambda c: float(c.stem.split("nme=")[1]))


def main() -> None:
    runs = sys.argv[1:] or sorted(d.name for d in CKPT_DIR.iterdir() if d.is_dir())

    ds = EarLandmarkDataset(split_csv=DATA_DIR / "test.csv", data_dir=DATA_DIR,
                            image_size=192)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    print(f"test split: {len(ds)} samples\n")

    results = {}
    for run in runs:
        ck = best_ckpt(run)
        if ck is None:
            print(f"{run:18s} (no checkpoint)")
            continue
        try:
            module = EarLandmarkerModule.load_from_checkpoint(str(ck))
            out = trainer.test(module, loader, verbose=False)[0]
            results[run] = out
            print(f"{run:18s} test/nme {out['test/nme']:.5f}  "
                  f"normal {out['test/nme_normal']:.5f}  "
                  f"tangential {out['test/nme_tangential']:.5f}  ({ck.name})")
        except Exception as e:
            print(f"{run:18s} ERR {type(e).__name__}: {e}")

    if results:
        best = min(results, key=lambda r: results[r]["test/nme"])
        print(f"\nbest: {best} at {results[best]['test/nme']:.5f}")


if __name__ == "__main__":
    main()
