"""Rank checkpoints by the score stored inside the file, not by their name.

The `nme=` field in a checkpoint filename is rounded to four decimals, and most
runs here end with two or three checkpoints tied at it. Ranking by that string
resolves the tie on readdir order -- which is epoch order -- and both `min()`
and a strict `<` comparison keep the FIRST element, so a filename-based
selector systematically returns the EARLIEST tied checkpoint rather than the
best one. On `fixed_conv_s42` that meant epoch 388 (0.014543440) in place of
epoch 433 (0.014519142).

That is a defect with a direction, not a coin flip, which is why this lives in
one place and the callers delegate to it.
"""
from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["checkpoint_score", "best_checkpoint"]


def checkpoint_score(path: str | Path) -> float:
    """Monitored score ModelCheckpoint stored in the file; +inf if unreadable.

    Falls back to the rounded `nme=` field only when the file cannot be loaded,
    so a corrupt checkpoint ranks last instead of raising.
    """
    path = Path(path)
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        for v in (ck.get("callbacks") or {}).values():
            if isinstance(v, dict) and v.get("current_score") is not None:
                return float(v["current_score"])
    except Exception:
        pass
    try:
        return float(path.stem.split("nme=")[1])
    except (IndexError, ValueError):
        return float("inf")


def best_checkpoint(run_dir: str | Path, recursive: bool = True) -> Path | None:
    """Lowest-scoring checkpoint under `run_dir`, or None if there are none.

    Considers only files carrying an `nme=` field, matching the previous
    behaviour of every caller; `last.ckpt` is excluded because it duplicates
    whichever epoch finished last and is not necessarily the best.
    """
    run_dir = Path(run_dir)
    globber = run_dir.rglob if recursive else run_dir.glob
    candidates = [c for c in globber("*.ckpt") if "nme=" in c.name]
    if not candidates:
        return None
    return min(candidates, key=checkpoint_score)
