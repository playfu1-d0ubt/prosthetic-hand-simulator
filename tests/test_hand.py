"""Tests for the hand drawing."""

import matplotlib.pyplot as plt

from hand import draw_hand


def test_draw_hand_adds_five_fingers_and_a_palm():
    """The hand should have five finger lines and one palm shape."""
    fig, ax = plt.subplots()
    draw_hand(ax)
    assert len(ax.lines) == 5
    assert len(ax.patches) == 1
    plt.close(fig)
