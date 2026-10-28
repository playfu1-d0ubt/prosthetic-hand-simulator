"""Train the gesture classifier, test it on new signals and print results."""

from classifier import evaluate, train
from compare import compare_scores

NOISE_FLOORS = (0.0, 0.3, 0.6, 1.0)


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


def print_comparison(noise_floors=NOISE_FLOORS, k=3):
    """Print both classifiers' percent right for each noise floor."""
    if len(noise_floors) == 0:
        raise ValueError("Invalid Value: noise_floors is empty")
    print("Noise floor  Typical-pair  KNN")
    for floor in noise_floors:
        typical_percent, knn_percent = compare_scores(floor, k)
        print(f"{floor:>11}  {typical_percent:>12.0f}%  {knn_percent:>3.0f}%")


def main():
    """Run the whole simulator."""
    typical = train()
    print("Typical window size per gesture:")
    for gesture, (size, roughness) in typical.items():
        print(f"  {gesture}: size {size:.2f}, roughness {roughness:.2f}")
    counts = evaluate(typical)
    print_table(counts)
    print()
    print_comparison()


if __name__ == "__main__":
    main()
