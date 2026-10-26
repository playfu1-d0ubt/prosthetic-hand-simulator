"""Tests for the gesture classifier."""

import numpy as np
import pytest

from classifier import evaluate, predict

TYPICAL = {
    "rest": np.array([0.08, 1.4]),
    "open": np.array([0.32, 0.6]),
    "fist": np.array([0.64, 0.4]),
}


def test_predict_picks_closest_gesture():
    """Each size should map to the gesture with the nearest typical size."""
    assert predict(np.array([0.10, 1.3]), TYPICAL) == "rest"
    assert predict(np.array([0.30, 0.65]), TYPICAL) == "open"
    assert predict(np.array([0.70, 0.35]), TYPICAL) == "fist"


def test_predict_empty_dict_raises():
    """An empty typical dict should raise ValueError."""
    with pytest.raises(ValueError):
        predict(np.array([0.5, 1.0]), {})


def test_evaluate_counts_every_window():
    """Each true gesture row should add up to the number of windows."""
    counts = evaluate(TYPICAL, seconds=2)
    for guesses in counts.values():
        assert sum(guesses.values()) == 10


def test_predict_wrong_shape_raises():
    """A single number instead of a pair should raise ValueError."""
    with pytest.raises(ValueError, match="shape"):
        predict(0.5, TYPICAL)
