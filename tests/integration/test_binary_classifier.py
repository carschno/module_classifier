import pytest
from src.module_classifier.classification.binary_classifier import MainEditionClassifier


class TestMainEditionClassifier:
    classifier = MainEditionClassifier.from_s3()

    @pytest.mark.parametrize(
        "texts,expected",
        [
            ([], []),
            ([""], [(False, pytest.approx(1.0, abs=0.001))]),
            (["test"], [(False, pytest.approx(1.0, abs=0.001))]),
        ],
    )
    def test_predict_texts(self, texts, expected):
        assert self.classifier.predict_texts(texts) == expected

    @pytest.mark.parametrize(
        "text,expected", [("", (False, pytest.approx(1.0, abs=0.001)))]
    )
    def test_predict_text(self, text, expected):
        assert self.classifier.predict_text(text) == expected
