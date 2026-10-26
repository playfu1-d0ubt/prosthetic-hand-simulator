"""Learn the typical window size of each gesture, then guess gestures."""

import numpy as np

from features import WINDOW_LENGTH, window_features
from signals import GESTURE_SIZE, make_signal


def train(seconds=20, seed=0, noise_floor=0.0):
    """Return a dict of gesture name to typical window size."""
    typical = {}
    for index, gesture in enumerate(GESTURE_SIZE):
        gesture_seed = seed + index
        signal = make_signal(gesture, seconds, gesture_seed, noise_floor=noise_floor)
        features = window_features(signal)
        typical[gesture] = features.mean(axis=0)
    return typical


def predict(features, typical):
    """Return the gesture whose typical size is closest to size."""
    if not typical:
        raise ValueError("No trained gestures. Run train() first.")
    if np.shape(features) != np.shape(next(iter(typical.values()))):
        raise ValueError("features must have the same shape as the typical values.")
    distances = {}
    for gesture in typical:
        distances[gesture] = np.linalg.norm(features - typical[gesture])
    return min(distances, key=distances.get)


def evaluate(
    typical, seconds=20, seed=100, window_length=WINDOW_LENGTH, noise_floor=0.0
):
    """Return counts[true_gesture][guessed_gesture] on new signals."""
    counts = {true: {guess: 0 for guess in typical} for true in typical}
    for index, true_gesture in enumerate(typical):
        gesture_seed = seed + index
        signal = make_signal(
            true_gesture, seconds, gesture_seed, noise_floor=noise_floor
        )
        features = window_features(signal, window_length)
        for feature in features:
            guess = predict(feature, typical)
            counts[true_gesture][guess] += 1
    return counts
