"""Tests for the k-nearest-neighbour guesser."""

import numpy as np
import pytest

from knn import predict_knn

EXAMPLES = np.array([[0.10], [0.30], [0.35], [0.62], [0.70]])
LABELS = ["rest", "open", "open", "fist", "fist"]
EMPTY = np.array([])
TEST_LABELS = ["rest", "open", "fist", "fist"]


def test_knn_known_answer():
    """The three closest examples to 0.45 are open, open and fist."""
    assert predict_knn(np.array([0.45]), EXAMPLES, LABELS, k=3) == "open"


def test_k_low():
    """k of less then 1 should raise a value error"""
    with pytest.raises(ValueError, match="k needs to"):
        predict_knn(np.array([0.45]), EXAMPLES, LABELS, k=0.1)


def test_empty():
    """Empty examples should raise a value error"""
    with pytest.raises(ValueError, match="examples is empty"):
        predict_knn(np.array([0.45]), EMPTY, LABELS, k=3)


def test_not_equal_label():
    """LABEL and EXAMPLES that do not equal should raise a value error"""
    with pytest.raises(ValueError, match="does not match"):
        predict_knn(np.array([0.45]), EXAMPLES, TEST_LABELS, k=3)


def test_k_high():
    """k greater then examples should raise a value error"""
    with pytest.raises(ValueError, match="k is greater"):
        predict_knn(np.array([0.45]), EXAMPLES, LABELS, k=6)
