"""Train the gesture classifier, test it on new signals and print results."""

from classifier import evaluate, train


def print_table(counts):
    """Print true gestures as rows and guessed gestures as columns."""
    names = list(counts)
    print(f"{'':>8}", end="")
    for name in names:
        print(f"{name:>8}", end="")
    print(f"{'right':>8}")
    for true_gesture, guesses in counts.items():
        print(f"{true_gesture:>8}", end="")
        for name in names:
            print(f"{guesses[name]:>8}", end="")
        total = sum(guesses.values())
        percent = 100 * guesses[true_gesture] / total
        print(f"{percent:>7.0f}%")


def main():
    """Run the whole simulator."""
    typical = train()
    print("Typical window size per gesture:")
    for gesture, size in typical.items():
        print(f"  {gesture}: {size:.2f}")
    counts = evaluate(typical)
    print_table(counts)


if __name__ == "__main__":
    main()
