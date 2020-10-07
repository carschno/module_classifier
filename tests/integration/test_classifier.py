import pytest

from src.module_classifier.classification.classifier import Classifier
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
            [('S6.M4', 0.5502018928527832)],
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
            [('S6.M5', 0.34261074662208557)],
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
            [('S6.M5', 0.34261074662208557), ('S1.M1', 0.30089762806892395)],
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
                ('S6.M8', 1.0000089406967163),
                ('S6.M3', 1.08975518742227e-05),
                ('S6.M9', 1.011734002531739e-05),
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