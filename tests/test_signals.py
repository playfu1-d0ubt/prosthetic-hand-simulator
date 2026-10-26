"""Tests for the fake signal generator."""

import numpy as np
import pytest

from signals import SAMPLE_RATE, make_signal


def test_signal_length():
    """Two seconds should give two seconds of readings."""
    signal = make_signal("fist", 2, seed=1)
    assert len(signal) == 2 * SAMPLE_RATE


def test_same_seed_same_signal():
    """The same seed should give exactly the same signal."""
    first = make_signal("open", 1, seed=5)
    second = make_signal("open", 1, seed=5)
    assert np.array_equal(first, second)


def test_unknown_gesture_raises():
    """A gesture name that does not exist should raise ValueError."""
    with pytest.raises(ValueError):
        make_signal("wave", 2)


def test_negative_noise_floor_raises():
    """A negative noise floor should raise our own ValueError."""
    with pytest.raises(ValueError, match="noise_floor"):
        make_signal("rest", 2, noise_floor=-1)


def test_signal_shorter_than_smoothing_raises():
    """A signal shorter than the smoothing width should raise ValueError."""
    with pytest.raises(ValueError, match="readings"):
        make_signal("fist", 0.01)
