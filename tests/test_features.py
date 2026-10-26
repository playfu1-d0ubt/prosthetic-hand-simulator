"""Tests for the window size feature."""

import numpy as np

from features import window_sizes


def test_window_sizes_known_answer():
    """Two hand-made windows should give the sizes worked out by hand."""
    signal = np.array([-0.2, 0.4, -0.3, 0.9, -0.7, 0.8])
    sizes = window_sizes(signal, 3)
    assert np.allclose(sizes, [0.3, 0.8])
