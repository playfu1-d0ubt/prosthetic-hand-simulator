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


def predict(size, typical):
    """Return the gesture whose typical size is closest to size."""
    if not typical:
        raise ValueError("No trained gestures. Run train() first.")
    distances = {}
    for gesture in typical:
        distances[gesture] = abs(size - typical[gesture])
    return min(distances, key=distances.get)


typical = {"rest": 0.08, "open": 0.32, "fist": 0.64}
print(predict(0.5, typical))
print(predict(0.2, typical))
print(predict(0.5, {}))
