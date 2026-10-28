"""Draw a simple hand that can open and close."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from signals import GESTURE_SIZE

PALM_WIDTH = 2.0
PALM_HEIGHT = 1.5

# (base x, base y, length) for thumb, index, middle, ring and little finger
FINGERS = (
    (-1.2, 0.6, 0.9),
    (-0.8, 1.5, 1.4),
    (-0.4, 1.5, 1.7),
    (0.0, 1.5, 1.5),
    (0.4, 1.5, 1.2),
)

GESTURE_OPENNESS = {
    "rest": 0.5,
    "open": 1.0,
    "fist": 0.1,
}


def openness_for(gesture):
    """Return how open the hand is for a gesture name."""
    if gesture not in GESTURE_OPENNESS:
        raise ValueError(
            f"Unknown gesture '{gesture}'. Use one of: {list(GESTURE_OPENNESS)}"
        )
    return GESTURE_OPENNESS[gesture]


def draw_hand(ax, openness=1.0):
    """Draw a palm and five fingers on ax based on openessed."""
    if openness > 1.0 or 0.0 > openness:
        raise ValueError(
            f"Invalid Value: openness is {openness}, openness need to be between 1.0 and 0.0"
        )
    palm = Rectangle((-PALM_WIDTH / 2, 0), PALM_WIDTH, PALM_HEIGHT, color="black")
    ax.add_patch(palm)
    for base_x, base_y, length in FINGERS:
        ax.plot(
            (base_x, base_x),
            (base_y, base_y + length * openness),
            color="black",
            linewidth=8,
        )
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3)
    fig.patch.set_facecolor("white")
    for ax, gesture in zip(axes, GESTURE_SIZE):
        draw_hand(ax, 0.0)
        ax.set_title(gesture, color="black")
    plt.show()
