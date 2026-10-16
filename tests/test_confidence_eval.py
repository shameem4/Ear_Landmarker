"""The confidence evaluation's helpers, and the artefact they exist to avoid.

`scripts/eval_confidence.py` concluded that heatmap concentration predicts
per-point error in domain, against a peer's null on renders. Two of its three
analyses would produce that conclusion spuriously if the helpers were wrong, so
they are pinned here:

  - a POOLED correlation can be carried by "landmark 7 is intrinsically hard and
    also intrinsically diffuse" rather than by anything about the choice a
    consumer makes, so the z-score must normalise PER LANDMARK;
  - selecting points by their OWN error truncates the correlated variable and
    drives rho toward 0 as a pure artefact, which would have manufactured
    exactly the null under test. The regime pass therefore selects whole images
    by mean error, and the artefact is demonstrated here so the reason the code
    is shaped that way survives in executable form.

Also covers guard_run.sh rejecting a unit-less memory cap, since a bare number
is read by systemd as BYTES and its instant OOM is indistinguishable from a
genuine memory blowup. That check lives in the shared wrap_work implementation
that scripts/guard_run.sh execs, so those two tests skip where it is absent.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """Import the script by path; `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "eval_confidence", ROOT / "scripts" / "eval_confidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ec():
    return _load()


# --- z-score ------------------------------------------------------------------

def test_zscore_normalises_per_landmark_not_globally(ec):
    """Two landmarks on wildly different scales must come out comparable.

    This is the whole point: a raw threshold over the pooled values would select
    landmark 1 entirely, because every one of its values is below every value of
    landmark 0.
    """
    conf = np.column_stack([
        np.array([10.0, 20.0, 30.0]),      # landmark 0, high scale
        np.array([0.1, 0.2, 0.3]),         # landmark 1, low scale
    ])
    z = ec.zscore_within_landmark(conf)

    assert np.allclose(z[:, 0], z[:, 1]), "same shape, different scale -> same z"
    assert np.allclose(z.mean(axis=0), 0.0)
    assert np.allclose(z.std(axis=0), 1.0)


def test_zscore_survives_a_constant_landmark(ec):
    """A landmark whose confidence never varies must not produce inf or nan.

    Real splits contain points the model always localises identically, and a
    division by a zero std would poison every pooled figure downstream.
    """
    conf = np.column_stack([np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0])])
    z = ec.zscore_within_landmark(conf)
    assert np.isfinite(z).all()
    assert np.allclose(z[:, 0], 0.0)


# --- within-landmark rho ------------------------------------------------------

def test_within_landmark_rho_finds_a_planted_monotone_relation(ec):
    """Confidence descending while error ascends is rho = -1 per landmark."""
    n, k = 40, 3
    conf = np.tile(np.linspace(1.0, 0.0, n)[:, None], (1, k))
    err = np.tile(np.linspace(0.0, 1.0, n)[:, None], (1, k))
    med, below, tot, lo, hi = ec.within_landmark_rho(conf, err)

    assert med == pytest.approx(-1.0)
    assert (below, tot) == (k, k)
    assert lo == pytest.approx(-1.0) and hi == pytest.approx(-1.0)


def test_within_landmark_rho_reports_no_signal_as_no_signal(ec):
    """Independent confidence and error must not be counted as below -0.1."""
    rng = np.random.default_rng(0)
    conf, err = rng.normal(size=(400, 10)), rng.normal(size=(400, 10))
    med, below, tot, _, _ = ec.within_landmark_rho(conf, err)

    assert abs(med) < 0.15, f"spurious correlation {med:+.3f} on independent data"
    assert tot == 10
    assert below <= 2, "independent columns should rarely clear -0.1"


def test_within_landmark_rho_refuses_too_few_rows(ec):
    """Below min_n nothing is scored, rather than a rho from 5 points."""
    conf, err = np.random.default_rng(1).normal(size=(5, 4)), \
        np.random.default_rng(2).normal(size=(5, 4))
    med, below, tot, _, _ = ec.within_landmark_rho(conf, err, min_n=30)
    assert tot == 0 and below == 0
    assert np.isnan(med)


# --- the artefact the regime pass is shaped to avoid --------------------------

def test_selecting_points_by_own_error_destroys_the_correlation(ec):
    """Why the regime pass selects IMAGES, not points.

    Plants a real relationship, then selects the worst points BY THEIR OWN
    ERROR. The correlation must collapse -- not because the signal is weak, but
    because the selection truncated the variable being correlated. Had
    eval_confidence.py selected this way, it would have reproduced the peer's
    null on its own data and "confirmed" the wrong conclusion.

    Noise is set so the full-sample rho lands near the -0.486 actually measured
    on photographs, not near -1. A near-deterministic planted relation survives
    truncation far better than a realistic one, so testing at -0.99 would
    understate the artefact and pick a threshold that means nothing.
    """
    rng = np.random.default_rng(42)
    n = 4000
    conf = rng.uniform(0.0, 1.0, size=(n, 1))
    err = (1.0 - conf) * 20.0 + rng.normal(0.0, 12.0, size=(n, 1))

    full, _, _, _, _ = ec.within_landmark_rho(conf, err)
    assert -0.60 < full < -0.35, (
        f"planted relation should sit near the measured -0.486, got {full:+.3f}")

    keep = (err[:, 0] > np.percentile(err[:, 0], 90))
    truncated, _, _, _, _ = ec.within_landmark_rho(conf[keep], err[keep])

    assert abs(truncated) < abs(full) / 2, (
        f"truncation should gut the correlation: full {full:+.3f}, "
        f"top-decile {truncated:+.3f}")


def test_selecting_whole_images_preserves_the_correlation(ec):
    """The same planted relation survives selection by MEAN IMAGE error.

    The complement of the test above: this is the selection the regime pass
    actually performs, and it must not manufacture a null.
    """
    rng = np.random.default_rng(7)
    n_img, k = 600, 12
    conf = rng.uniform(0.0, 1.0, size=(n_img, k))
    # A per-image difficulty offset, so images differ in overall error and
    # there is something for a per-image selection to bite on.
    offset = rng.uniform(0.0, 10.0, size=(n_img, 1))
    err = (1.0 - conf) * 20.0 + offset + rng.normal(0.0, 1.0, size=(n_img, k))

    full, _, _, _, _ = ec.within_landmark_rho(conf, err)
    worst = np.argsort(-err.mean(axis=1))[:100]
    sub, _, _, _, _ = ec.within_landmark_rho(conf[worst], err[worst])

    assert sub < -0.5, f"per-image selection should keep the signal: {sub:+.3f}"
    assert abs(sub - full) < 0.4, (
        f"per-image selection should not move rho much: {full:+.3f} -> {sub:+.3f}")


# --- guard_run.sh units -------------------------------------------------------

# scripts/guard_run.sh is a thin wrapper that execs a shared implementation in
# wrap_work/, which is NOT version-controlled and sits outside this repo. So
# these two tests depend on a file a fresh clone will not have: skipped when it
# is absent, because a missing dependency is not a finding and an exit 127 here
# would read as the unit check having broken.
GUARD = ROOT / "scripts" / "guard_run.sh"
IMPL = Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/tools/guard_run.sh")
needs_impl = pytest.mark.skipif(
    not IMPL.is_file(),
    reason=f"shared guard_run implementation not present at {IMPL}")


def _guard(mem: str):
    return subprocess.run(
        ["bash", str(GUARD), mem, "30", "/dev/null", "/bin/true"],
        capture_output=True, text=True, timeout=60)


@needs_impl
@pytest.mark.parametrize("mem", ["12", "28", "4"])
def test_guard_run_rejects_a_unitless_memory_cap(mem):
    """A bare number is BYTES to systemd; the job dies before printing anything.

    Rejected with a non-zero exit and a message naming the fix, because the
    symptom it otherwise produces -- an empty log and "KILLED (memory cap 12)"
    -- reads as a genuine blowup and sends you after the wrong cause.
    """
    r = _guard(mem)
    assert r.returncode == 2, f"unit-less {mem!r} should be refused"
    assert "no unit" in r.stderr
    assert f"{mem}G" in r.stderr, "the message should name the corrected value"


@needs_impl
@pytest.mark.parametrize("mem", ["12G", "500M", "infinity"])
def test_guard_run_accepts_a_united_memory_cap(mem):
    r = _guard(mem)
    assert r.returncode == 0, f"{mem!r} should be accepted: {r.stderr}"
