"""Make fake muscle signals for each gesture."""

import numpy as np

SAMPLE_RATE = 1000  # readings per second

GESTURE_SIZE = {
    "rest": 0.1,
    "open": 0.4,
    "fist": 0.8,
}


def make_signal(gesture, seconds, seed=None, noise_floor=0.0):
    """Return a fake muscle signal as a numpy array."""
    n_readings = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(seed)
    if gesture not in GESTURE_SIZE:
        raise ValueError(
            f"Unknown gesture '{gesture}'. Use one of: {list(GESTURE_SIZE)}"
        )
    if noise_floor < 0:
        raise ValueError(f"noise_floor must be 0 or more, got {noise_floor}.")
    spread = GESTURE_SIZE[gesture]
    signal = rng.normal(0, spread, n_readings)
    return signal + rng.normal(0, noise_floor, n_readings)
