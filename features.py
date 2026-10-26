"""Turn a muscle signal into one size number per window."""

import numpy as np

from signals import SAMPLE_RATE

WINDOW_LENGTH = int(0.2 * SAMPLE_RATE)  # readings per window, 0.2 * SAMPLE_RATE


def split_windows(signal, window_length):
    """Cut a signal into rows, one row per whole window."""
    if window_length < 1:
        raise ValueError(f"Window length must be at least 1, got {window_length}.")
    n_windows = len(signal) // window_length
    if n_windows == 0:
        raise ValueError(
            f"Signal has {len(signal)} readings, need at least {window_length}."
        )
    trimmed = signal[: n_windows * window_length]
    return trimmed.reshape(n_windows, window_length)


def window_sizes(signal, window_length=WINDOW_LENGTH):
    """Return the average absolute value of each window."""
    rows = split_windows(signal, window_length)
    return np.abs(rows).mean(axis=1)


def window_roughness(signal, window_length=WINDOW_LENGTH):
    """Return the roughness of each window: average jump divided by size."""
    if window_length < 2:
        raise ValueError(
            f"Roughness needs a window of at least 2, got {window_length}."
        )
    rows = split_windows(signal, window_length)
    jumps = np.diff(rows, axis=1)
    avg_jump = np.abs(jumps).mean(axis=1)
    avg_size = np.abs(rows).mean(axis=1)
    return avg_jump / avg_size
