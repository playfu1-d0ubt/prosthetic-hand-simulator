"""Draw charts of window sizes for each gesture."""

import matplotlib.pyplot as plt
import numpy as np

from features import window_sizes
from signals import GESTURE_SIZE, make_signal


def plot_sizes(ax, window_length, seconds=20):
    """Draw one histogram of window sizes per gesture on ax."""
    bins = np.linspace(0, 1.6, 40)
    for index, gesture in enumerate(GESTURE_SIZE):
        gesture_seed = 200 + index
        signal = make_signal(gesture, seconds, gesture_seed)
        sizes = window_sizes(signal, window_length)
        ax.hist(sizes, bins=bins, alpha=0.6, label=gesture)
    ax.set_title(f"Window length {window_length}")
    ax.set_xlabel("Window size")
    ax.set_ylabel("Number of windows")
    ax.legend()


def make_comparison_figure():
    """Return a figure comparing a long window with a short window."""
    fig, (ax_long, ax_short) = plt.subplots(1, 2, figsize=(10, 4))
    plot_sizes(ax_long, 200)
    plot_sizes(ax_short, 5)
    fig.tight_layout()
    return fig


def main():
    """Save the comparison chart for the README."""
    fig = make_comparison_figure()
    fig.savefig("export/window_comparison.png", dpi=150, facecolor="white")
    print("Saved export/window_comparison.png")


if __name__ == "__main__":
    main()
