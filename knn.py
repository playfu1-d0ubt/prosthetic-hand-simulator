"""Guess a gesture from the k closest labelled example windows."""

from collections import Counter

import numpy as np


def predict_knn(features, examples, labels, k=3):
    """Return the most common label among the k closest examples."""
    if k < 1:
        raise ValueError(f"Invaild Value: k is {k}, k needs to be greater then 1")
    if len(examples) == 0:
        raise ValueError("Invaild Value: examples is empty")
    if len(labels) != len(examples):
        raise ValueError("Invaild Value: labels does not match examples")
    if k > len(examples):
        raise ValueError("Invaild Value: k is greater then examples")
    distances = np.linalg.norm(examples - features, axis=1)
    order = np.argsort(distances)
    votes = [labels[i] for i in order[:k]]
    return Counter(votes).most_common(1)[0][0]
