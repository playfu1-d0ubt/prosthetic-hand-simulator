"""Tests for the gesture classifier."""

from classifier import predict

TYPICAL = {"rest": 0.08, "open": 0.32, "fist": 0.64}


def test_predict_picks_closest_gesture():
    """Each size should map to the gesture with the nearest typical size."""
    assert predict(0.10, TYPICAL) == "rest"
    assert predict(0.30, TYPICAL) == "open"
    assert predict(0.70, TYPICAL) == "fist"
