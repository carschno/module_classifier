import pytest

from src.module_classifier.classification.module_classifier import ModuleClassifier


def test_init():
    with pytest.raises(ValueError):
        ModuleClassifier("/does/not/exist")
