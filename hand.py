"""Draw a simple hand that can open and close."""

from matplotlib.patches import Rectangle

PALM_WIDTH = 2.0
PALM_HEIGHT = 1.5

# (base x, base y, length) for thumb, index, middle, ring and little finger
FINGERS = (
    (-1.0, 0.5, 0.9),
    (-0.8, 1.5, 1.4),
    (-0.4, 1.5, 1.7),
    (0.0, 1.5, 1.5),
    (0.4, 1.5, 1.2),
)


def draw_hand(ax):
    """Draw a palm and five fingers on ax."""
    palm = Rectangle((-PALM_WIDTH / 2, 0), PALM_WIDTH, PALM_HEIGHT, color="black")
    ax.add_patch(palm)
    for base_x, base_y, length in FINGERS:
        ax.plot((base_x, base_x), (base_y, base_y + length), color="black", linewidth=8)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    draw_hand(ax)
    plt.show()
