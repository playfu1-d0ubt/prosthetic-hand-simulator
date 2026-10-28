"""Tests for the demo signal and openness values."""

import numpy as np
import pytest

from classifier import train
from demo import make_demo_signal, openness_per_window
from hand import GESTURE_OPENNESS


def test_one_openness_value_per_window():
    """A 3 gesture, 2 second signal should give 30 openness values."""
    values = openness_per_window(make_demo_signal(), train())
    assert len(values) == 30


def test_values_are_known_openness_numbers():
    """Every value should be one of the openness numbers in the hand."""
    values = openness_per_window(make_demo_signal(), train())
    assert set(values) <= set(GESTURE_OPENNESS.values())


def test_empty_signal_raises():
    """Empty signal should raise a value error"""
    with pytest.raises(ValueError):
        openness_per_window(np.array([]), train())
