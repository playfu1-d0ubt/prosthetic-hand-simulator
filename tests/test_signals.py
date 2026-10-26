"""Tests for the fake signal generator."""

from signals import SAMPLE_RATE, make_signal


def test_signal_length():
    """Two seconds should give two seconds of readings."""
    signal = make_signal("fist", 2, seed=1)
    assert len(signal) == 2 * SAMPLE_RATE
