import pytest

from src.module_classifier.classification.binary_classifier import BinaryClassifier


class TestBinaryClassifier:
    @pytest.mark.parametrize(
        "label,expected", [("__label__True", True), ("__label__False", False)]
    )
    def test_deserialize_label(self, label, expected):
        assert BinaryClassifier._deserialize_label(label) == expected
