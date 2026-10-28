"""Tests for the printed output in main.py."""

import pytest

from main import print_comparison


def test_print_comparison_prints_header(capsys):
    """The table should start with a header line."""
    print_comparison((0.0,))
    output = capsys.readouterr().out
    assert "Noise floor" in output


def test_print_comparison_empty():
    """empty noise floor should raise a ValueError"""
    with pytest.raises(ValueError):
        print_comparison(())
