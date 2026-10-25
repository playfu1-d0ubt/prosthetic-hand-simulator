"""Learn the typical window size of each gesture, then guess gestures."""

from features import window_sizes
from signals import GESTURE_SIZE, make_signal


def train(seconds=20, seed=0):
    """Return a dict of gesture name to typical window size."""
    typical = {}
    for index, gesture in enumerate(GESTURE_SIZE):
        gesture_seed = seed + index
        signal = make_signal(gesture, seconds, gesture_seed)
        sizes = window_sizes(signal)
        typical[gesture] = sizes.mean()
    return typical
