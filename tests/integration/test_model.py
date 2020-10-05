import pytest

from src.module_classifier.classification.model import Model
from ..conftest import TEST_MODEL, does_not_raise


@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
def test_init():
    assert isinstance(Model(TEST_MODEL), Model)


@pytest.mark.parametrize(
    "row,k,expected,exception",
    [
        ({}, 1, [], pytest.raises(ValueError)),
        ({"title": ""}, 1, [], pytest.raises(ValueError)),
        (
            {
                "item_title": "",
                "authors": "",
                "publication_name": "",
                "abstract_description": "",
            },
            1,
            [('__label__S6.M1', 1.0000098943710327)],
            does_not_raise(),
        ),
        (
            {
                "item_title": "test title",
                "authors": "test author",
                "publication_name": "test publication",
                "abstract_description": "test abstract",
            },
            1,
            [('__label__S1.M1', 0.3548894226551056)],
            does_not_raise(),
        ),
        (
            {
                "item_title": "test title",
                "authors": "test author",
                "publication_name": "test publication",
                "abstract_description": "test abstract",
            },
            2,
            [
                ('__label__S1.M1', 0.3548894226551056),
                ('__label__S1.M12', 0.21194346249103546),
            ],
            does_not_raise(),
        ),
    ],
    ids=[
        "empty input",
        "missing fields",
        "empty fields",
        "test fields",
        "k=2",
    ],
)
@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
def test_predict_row(row, k, expected, exception):
    model = Model(TEST_MODEL)
    with exception:
        assert model.predict_row(row, k) == expected


@pytest.mark.parametrize(
    "text,k,expected",
    [
        (
            "ai automation",
            3,
            [
                ('__label__S6.M8', 1.0000026226043701),
                ('__label__S6.M3', 1.4651314813818317e-05),
                ('__label__S6.M9', 1.1984460797975771e-05),
            ],
        )
    ],
    ids=["AI & Automation"],
)
@pytest.mark.skipif(TEST_MODEL is None, reason="'TEST_MODEL' not specified.")
def test_predict_text(text, k, expected):
    model = Model(TEST_MODEL)
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
        assert isinstance(Model.download(remote, local), Model)
