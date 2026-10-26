"""Tests for the fake signal generator."""

import numpy as np

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
