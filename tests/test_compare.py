"""Tests for the classifier comparison."""

import pytest

from classifier import train
from compare import make_dataset, percent_right


def test_dataset_shapes_match():
    """Every feature row should have exactly one label."""
    features, labels = make_dataset(seconds=2)
    assert len(features) == len(labels)


def test_percent_right_in_range():
    """Percent right should be between 0 and 100."""
    typical = train()
    features, labels = make_dataset(seed=100)
    result = percent_right(typical, features, labels)
    assert 0 <= result <= 100


def test_percent_right_empty_labels():
    """Empty labels should raise a ValueError."""
    typical = train()
    with pytest.raises(ValueError):
        percent_right(typical, [], [])


def test_percent_right_length_mismatch():
    """Features and labels of different lengths should raise a ValueError."""
    typical = train()
    features, labels = make_dataset(seed=100)
    with pytest.raises(ValueError):
        percent_right(typical, features, labels[:-1])
