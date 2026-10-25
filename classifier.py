"""Learn the typical window size of each gesture, then guess gestures."""

from features import WINDOW_LENGTH, window_sizes
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


def predict(size, typical):
    """Return the gesture whose typical size is closest to size."""
    if not typical:
        raise ValueError("No trained gestures. Run train() first.")
    distances = {}
    for gesture in typical:
        distances[gesture] = abs(size - typical[gesture])
    return min(distances, key=distances.get)


def evaluate(typical, seconds=20, seed=100, window_length=WINDOW_LENGTH):
    """Return counts[true_gesture][guessed_gesture] on new signals."""
    counts = {true: {guess: 0 for guess in typical} for true in typical}
    for index, true_gesture in enumerate(typical):
        gesture_seed = seed + index
        signal = make_signal(true_gesture, seconds, gesture_seed)
        sizes = window_sizes(signal, window_length)
        for size in sizes:
            guess = predict(float(size), typical)
            counts[true_gesture][guess] += 1
    return counts
