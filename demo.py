"""Turn a changing muscle signal into hand openness values."""

import numpy as np

from classifier import predict, train
from features import window_features
from hand import openness_for
from signals import GESTURE_SIZE, make_signal


def make_demo_signal(seconds_each=2, seed=300, noise_floor=0.0):
    """Return one signal that runs through every gesture in turn."""
    pieces = []
    for index, gesture in enumerate(GESTURE_SIZE):
        signal = make_signal(
            gesture, seconds_each, seed + index, noise_floor=noise_floor
        )
        pieces.append(signal)
    return np.concatenate(pieces)


def openness_per_window(signal, typical):
    """Return a list with one openness value per window of the signal."""
    features = window_features(signal)
    values = []
    for row in features:
        guess = predict(row, typical)
        values.append(openness_for(guess))
    return values
