"""Make fake muscle signals for each gesture."""

import numpy as np

SAMPLE_RATE = 1000  # readings per second

GESTURE_SIZE = {
    "rest": 0.1,
    "open": 0.4,
    "fist": 0.8,
}


def make_signal(gesture, seconds, seed=None):
    """Return a fake muscle signal as a numpy array."""
    n_reading = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(seed)
    if gesture not in GESTURE_SIZE:
        raise ValueError(
            f"Unknown gesture '{gesture}'. Use one of: {list(GESTURE_SIZE)}"
        )
    spread = GESTURE_SIZE[gesture]
    return rng.normal(0, spread, n_reading)
