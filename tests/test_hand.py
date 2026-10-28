"""Tests for the hand drawing."""

import matplotlib.pyplot as plt
import pytest

from hand import GESTURE_OPENNESS, GESTURE_SIZE, draw_hand, openness_for


def test_draw_hand_adds_five_fingers_and_a_palm():
    """The hand should have five finger lines and one palm shape."""
    fig, ax = plt.subplots()
    draw_hand(ax)
    assert len(ax.lines) == 5
    assert len(ax.patches) == 1
    plt.close(fig)


def test_gesture_names_match_signals():
    """The hand and the signal maker should know the same gestures."""
    assert set(GESTURE_OPENNESS) == set(GESTURE_SIZE)


def test_openness_for_invaild_gesture():
    """A gesture name that does not exist should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown gesture"):
        openness_for("wave")


def test_draw_hand_invaild_openness():
    """Openness outside the range shouls raise ValueError."""
    with pytest.raises(ValueError, match="between 1.0 and 0.0"):
        fig, ax = plt.subplots()
        draw_hand(ax, 1.5)
        plt.close(fig)
