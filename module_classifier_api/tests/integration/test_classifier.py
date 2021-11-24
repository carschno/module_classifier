import pytest
from src.module_classifier.classification.classifier import Classifier
from src.module_classifier.models import Module

from ..conftest import TEST_MODEL, does_not_raise


@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
def test_init():
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
            [(Module(section=3, module=6), 1.0000100135803223)],
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
            [(Module(section=3, module=3), 0.4488534927368164)],
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
                (Module(section=3, module=3), 0.4488534927368164),
                (Module(section=6, module=7), 0.4234696626663208),
            ],
        ),
    ],
    ids=["empty fields", "test fields", "k=2"],
)
@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
def test_predict_row(row, columns, k, expected):
    model = Classifier(TEST_MODEL)
    assert model.predict_row(row, k, columns=columns) == expected


@pytest.mark.parametrize(
    "text,k,expected",
    [
        (
            "ai automation",
            3,
            [
                (Module(section=6, module=8), 1.0000100135803223),
                (Module(section=4, module=5), 1.0000003385357559e-05),
                (Module(section=4, module=2), 1.0000003385357559e-05),
            ],
        )
    ],
    ids=["AI & Automation"],
)
@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
def test_predict_text(text, k, expected):
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
@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
def test_download(remote, local, expected_exception):
    with expected_exception:
        assert isinstance(Classifier.download(remote, local), Classifier)
