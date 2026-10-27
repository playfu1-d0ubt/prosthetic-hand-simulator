"""Tests for the classifier comparison."""

from compare import make_dataset


def test_dataset_shapes_match():
    """Every feature row should have exactly one label."""
    features, labels = make_dataset(seconds=2)
    assert len(features) == len(labels)
