import pytest

from src.module_classifier.classification.classifier import Classifier


def test_init():
    with pytest.raises(ValueError):
        Classifier("/does/not/exist")
