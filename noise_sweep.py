"""Show how classifier accuracy falls as background noise rises."""

from classifier import evaluate, train

NOISE_FLOORS = (0.0, 0.3, 0.6, 1.0)


def accuracy(counts):
    """Return the percent of windows guessed right."""
    right = sum(counts[gesture][gesture] for gesture in counts)
    total = sum(sum(row.values()) for row in counts.values())
    return 100 * right / total


def main():
    """Print accuracy for each noise floor."""
    print("Noise floor  Accuracy")
    for floor in NOISE_FLOORS:
        typical = train(noise_floor=floor)
        counts = evaluate(typical, noise_floor=floor)
        print(f"{floor:>11}  {accuracy(counts):>7.0f}%")


if __name__ == "__main__":
    main()
