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
            [("S4.M10", 0.9931851029396057)],
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
            [("S6.M1", 0.5474758148193359)],
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
            [("S6.M1", 0.5474758148193359), ("S6.M5", 0.3911016881465912)],
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
                ("S6.M8", 1.0000100135803223),
                ("S6.M6", 1.0011730410042219e-05),
                ("S6.M3", 1.0000422662415076e-05),
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
