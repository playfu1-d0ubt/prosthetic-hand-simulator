"""Tests for the window size feature."""

import numpy as np
import pytest

from features import window_roughness, window_sizes


def test_window_sizes_known_answer():
    """Two hand-made windows should give the sizes worked out by hand."""
    signal = np.array([-0.2, 0.4, -0.3, 0.9, -0.7, 0.8])
    sizes = window_sizes(signal, 3)
    assert np.allclose(sizes, [0.3, 0.8])


def test_too_short_signal_raises():
    """A signal shorter than one window should raise ValueError."""
    signal = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        window_sizes(signal, 3)


def test_window_roughness_known_answer():
    """A zigzag window and a flat window should give 2.0 and 0.0."""
    signal = np.array([1, -1, 1, -1, 1, 1, 1, 1])
    assert np.allclose(window_roughness(signal, 4), [2.0, 0.0])


def test_roughness_window_of_one_raises():
    """A window of 1 has no neighbours, so it should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2"):
        window_roughness(np.array([1.0, 2.0, 3.0]), 1)
