import pytest
from src.module_classifier.classification import Classifier, Prediction
from src.module_classifier.preprocessing import Module

from ..conftest import TEST_MODEL, does_not_raise


@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
class TestClassifier:
    def test_init(self):
        assert isinstance(Classifier(TEST_MODEL), Classifier)

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
        model = Classifier(TEST_MODEL)
        assert model.predict_row(row, k, columns=columns) == expected

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
        model = Classifier(TEST_MODEL)
        assert model.predict_text(text, k) == expected

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
        assert len(Classifier(TEST_MODEL).labels) == 60

    def test_modules(self):
        model = Classifier(TEST_MODEL)
        assert len(model.modules) == 60
        assert all(isinstance(m, Module) for m in model.modules)
