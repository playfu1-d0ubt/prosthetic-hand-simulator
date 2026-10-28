"""Turn a changing muscle signal into hand openness values."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from classifier import predict, train
from features import window_features
from hand import draw_hand, openness_for
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


def make_animation(fig, ax, values, interval=200):
    """Return an animation of the hand opening and closing through values."""
    if len(values) == 0:
        raise ValueError("Invaild value: values is empty")

    def update(frame):
        """Draw the hand for one window."""
        ax.clear()
        draw_hand(ax, values[frame])
        ax.set_title(f"Window {frame + 1} of {len(values)}", color="black")

    return FuncAnimation(fig, update, frames=len(values), interval=interval)


def main():
    """Show the hand reacting to the demo signal."""
    typical = train()
    values = openness_per_window(make_demo_signal(), typical)
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    anim = make_animation(fig, ax, values)
    anim.save("export/hand_demo.gif", writer="pillow", fps=5)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
