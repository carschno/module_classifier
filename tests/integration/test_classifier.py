import numpy as np
import pytest
from src.module_classifier.classification import Classifier, Prediction, Predictions
from src.module_classifier.preprocessing import Module

from ..conftest import TEST_MODEL, does_not_raise


@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
class TestClassifier:
    classifier = Classifier(TEST_MODEL)

    def test_init(self):
        assert isinstance(self.classifier, Classifier)

    @pytest.mark.parametrize(
        "row,columns,k,expected",
        [
            (
                {
                    "item_title": "",
                    "authors": "",
                    "publication_name": "",
                    "abstract_description": "",
                },
                (
                    "item_title",
                    "authors",
                    "publication_name",
                    "abstract_description",
                ),
                1,
                [Prediction(Module(section=3, module=6), 1.0000100135803223)],
            ),
            (
                {
                    "item_title": "test title",
                    "authors": "test author",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                },
                (
                    "item_title",
                    "authors",
                    "publication_name",
                    "abstract_description",
                ),
                1,
                [Prediction(Module(section=3, module=3), 0.4488534927368164)],
            ),
            (
                {
                    "item_title": "test title",
                    "authors": "test author",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                },
                (
                    "item_title",
                    "authors",
                    "publication_name",
                    "abstract_description",
                ),
                2,
                [
                    Prediction(Module(section=3, module=3), 0.4488534927368164),
                    Prediction(Module(section=6, module=7), 0.4234696626663208),
                ],
            ),
        ],
        ids=["empty fields", "test fields", "k=2"],
    )
    def test_predict_row(self, row, columns, k, expected):
        assert self.classifier.predict_row(row, k, columns=columns) == expected

    @pytest.mark.parametrize(
        "rows, columns, k, expected",
        [
            ([], [], 1, []),
            (
                [
                    {"text": "a text about China"},
                    {"text": "something related to artificial intelligence"},
                ],
                ["text"],
                1,
                [
                    Predictions(
                        labels=["__label__S5_M8"],
                        probs=np.array([0.99993765], dtype=np.float32),
                    ),
                    Predictions(
                        labels=["__label__S6_M8"],
                        probs=np.array([1.0000085], dtype=np.float32),
                    ),
                ],
            ),
        ],
    )
    def test_predict_rows(self, rows, columns, k, expected):
        assert self.classifier.predict_rows(rows, k, columns=columns) == expected

    @pytest.mark.parametrize(
        "text,k,expected",
        [
            (
                "ai automation",
                3,
                [
                    Prediction(Module(section=6, module=8), 1.0000100135803223),
                    Prediction(Module(section=4, module=5), 1.0000003385357559e-05),
                    Prediction(Module(section=4, module=2), 1.0000003385357559e-05),
                ],
            )
        ],
        ids=["AI & Automation"],
    )
    def test_predict_text(self, text, k, expected):
        assert self.classifier.predict_text(text, k) == expected

    @pytest.mark.parametrize(
        "texts,k,expected",
        [
            ([], 3, []),
            (
                ["text about china", "ai automation"],
                3,
                [
                    Predictions(
                        labels=["__label__S5_M8", "__label__S6_M9", "__label__S6_M8"],
                        probs=np.array([9.9993765e-01, 7.7751087e-05, 1.4639512e-05]),
                    ),
                    Predictions(
                        labels=["__label__S6_M8", "__label__S4_M5", "__label__S4_M2"],
                        probs=np.array([1.0000100e00, 1.0000003e-05, 1.0000003e-05]),
                    ),
                ],
            ),
        ],
    )
    def test_predict_texts(self, texts, k, expected):
        for prediction, e in zip(self.classifier.predict_texts(texts, k), expected):
            assert prediction.labels == e.labels
            assert np.allclose(prediction.probs, e.probs)

    @pytest.mark.parametrize(
        "remote,local,expected_exception",
        [
            ("", TEST_MODEL, does_not_raise()),
            ("", "", pytest.raises(NotImplementedError)),
        ],
        ids=["test model", "download"],
    )
    def test_download(self, remote, local, expected_exception):
        with expected_exception:
            assert isinstance(Classifier.download(remote, local), Classifier)

    def test_labels(self):
        assert len(self.classifier.labels) == 60

    def test_modules(self):
        assert len(self.classifier.modules) == 60
        assert all(isinstance(m, Module) for m in self.classifier.modules)
