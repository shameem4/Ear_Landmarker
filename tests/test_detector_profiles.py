"""The detector gate is a consumer choice, so the name must resolve to the
number and the default must not have moved."""
import pytest

from inference import (DETECTOR_CONFIDENCE, DETECTOR_PROFILES,
                       DETECTOR_PROFILE_DEFAULT, resolve_confidence)


def test_default_profile_is_the_shipped_value():
    # Naming the choice must not have changed shipped behaviour.
    assert DETECTOR_CONFIDENCE == 0.70
    assert DETECTOR_PROFILES[DETECTOR_PROFILE_DEFAULT] == DETECTOR_CONFIDENCE


def test_none_resolves_to_the_default_profile():
    assert resolve_confidence(None) == DETECTOR_PROFILES[DETECTOR_PROFILE_DEFAULT]
    assert resolve_confidence() == DETECTOR_CONFIDENCE


@pytest.mark.parametrize("name,gate", sorted(DETECTOR_PROFILES.items()))
def test_names_resolve_to_their_gate(name, gate):
    assert resolve_confidence(name) == gate


def test_profiles_are_ordered_by_how_few_detections_they_emit():
    # The ordering is about detection COUNT, not quality: an FP audit found 64
    # of 64 sampled false positives were real unlabelled ears, so a higher gate
    # buys fewer detections rather than better ones.
    assert (DETECTOR_PROFILES["interactive"]
            < DETECTOR_PROFILES["fewer"]
            < DETECTOR_PROFILES["fewest"])


def test_default_is_the_most_permissive_profile():
    # Nothing supports a higher gate on quality grounds, so the default must
    # not be one of the recall-sacrificing settings.
    assert DETECTOR_PROFILES[DETECTOR_PROFILE_DEFAULT] == min(DETECTOR_PROFILES.values())


def test_float_passes_through():
    assert resolve_confidence(0.83) == pytest.approx(0.83)
    assert resolve_confidence(1) == pytest.approx(1.0)


def test_unknown_name_names_the_alternatives():
    with pytest.raises(ValueError) as e:
        resolve_confidence("aggressive")
    msg = str(e.value)
    assert "aggressive" in msg
    for name in DETECTOR_PROFILES:
        assert name in msg


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_out_of_range_gate_is_refused(bad):
    # A gate of 0 admits every box; above 1 admits none. Both are silent
    # misconfigurations rather than errors if they are allowed through.
    with pytest.raises(ValueError):
        resolve_confidence(bad)


def _parse(argv):
    """Parse argv with main()'s own parser, without running inference.

    main() builds the parser inline, so the namespace is captured by letting
    parse_args run and then bailing out before any model loads.
    """
    import argparse
    import unittest.mock as mock

    import inference

    class _Stop(Exception):
        pass

    real = argparse.ArgumentParser.parse_args
    got = {}

    def grab(self, args=None, namespace=None):
        got["ns"] = real(self, argv, namespace)
        raise _Stop

    with mock.patch.object(argparse.ArgumentParser, "parse_args", grab):
        with pytest.raises(_Stop):
            inference.main()
    return got["ns"]


def test_cli_takes_a_profile_or_a_number_but_not_both():
    # Letting one silently win is how a caller ends up running a gate they did
    # not pick, so argparse should refuse the pair outright.
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        _parse(["image", "x.jpg", "--profile", "fewest", "--confidence", "0.8"])

    named = _parse(["image", "x.jpg", "--profile", "fewest"])
    assert named.profile == "fewest" and named.confidence is None

    numeric = _parse(["image", "x.jpg", "--confidence", "0.8"])
    assert numeric.confidence == 0.8 and numeric.profile is None


def test_cli_default_leaves_both_unset_so_the_default_profile_applies():
    ns = _parse(["image", "x.jpg"])
    assert ns.profile is None and ns.confidence is None
    chosen = ns.confidence if ns.confidence is not None else ns.profile
    assert resolve_confidence(chosen) == DETECTOR_CONFIDENCE
