"""Compare the typical-pair classifier with k-nearest-neighbour."""

import numpy as np

from classifier import predict, train
from features import window_features
from knn import predict_knn
from signals import GESTURE_SIZE, make_signal


def make_dataset(seconds=20, seed=0, noise_floor=0.0):
    """Return (features, labels) with one row per window of every gesture."""
    all_features = []
    all_labels = []
    for index, gesture in enumerate(GESTURE_SIZE):
        gesture_seed = seed + index
        signal = make_signal(gesture, seconds, gesture_seed, noise_floor=noise_floor)
        features = window_features(signal)
        all_features.extend(features)
        all_labels.extend([gesture] * len(features))
    return np.vstack(all_features), all_labels


def percent_right(typical, features, labels):
    """Return the percent of windows the typical-pair classifier guesses right."""
    if len(labels) == 0:
        raise ValueError("labels is empty, nothing to score.")
    if len(labels) != len(features):
        raise ValueError("Invaild Value: labels and features do not match")
    right = 0
    for row, label in zip(features, labels):
        guess = predict(row, typical)
        if guess == label:
            right += 1
    return 100 * right / len(labels)


def percent_right_knn(train_features, train_labels, test_features, test_labels, k=3):
    """Return the percent of test windows k-nearest-neighbour guesses right."""
    if len(test_labels) == 0:
        raise ValueError("Invalid Value: test_labels is empty")
    if len(test_labels) != len(test_features):
        raise ValueError("Invalid Value: test_labels and test_features do not match")
    right = 0
    for row, label in zip(test_features, test_labels):
        guess = predict_knn(row, train_features, train_labels, k)
        if guess == label:
            right += 1
    return 100 * right / len(test_labels)


def compare_scores(noise_floor=0.0, k=3):
    """Return (typical-pair percent, KNN percent) on the same test windows."""
    typical = train(noise_floor=noise_floor)
    train_features, train_labels = make_dataset(seed=0, noise_floor=noise_floor)
    test_features, test_labels = make_dataset(seed=100, noise_floor=noise_floor)
    typical_percent = percent_right(typical, test_features, test_labels)
    knn_percent = percent_right_knn(
        train_features, train_labels, test_features, test_labels, k
    )
    return typical_percent, knn_percent
