"""Tests for the demo signal and openness values."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from classifier import train
from demo import make_animation, make_demo_signal, openness_per_window
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


def test_make_animation_returns_animation():
    """The animation should be a FuncAnimation object."""
    fig, ax = plt.subplots()
    anim = make_animation(fig, ax, [0.5, 1.0, 0.1])
    assert isinstance(anim, FuncAnimation)
    fig.canvas.draw()
    plt.close(fig)


def test_empty_values_raises():
    """Empty values should raise a value error"""
    with pytest.raises(ValueError):
        fig, ax = plt.subplots()
        make_animation(fig, ax, [])
    plt.close(fig)
