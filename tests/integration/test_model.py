import pytest

from src.module_classifier.classification.model import Model
from ..conftest import TEST_MODEL, does_not_raise


@pytest.mark.parametrize(
    "model_path,expected_exception",
    [
        ("/does/not/exist", pytest.raises(ValueError)),
        (TEST_MODEL, does_not_raise()),
    ],
    ids=["Model file does not exist", "Model exists"],
)
def test_init(model_path, expected_exception):
    with expected_exception:
        Model(model_path)


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
            [('__label__S6.M5', 1.0000098943710327)],
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
            [('__label__S6.M5', 0.6045941710472107)],
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
                ('__label__S6.M5', 0.6045941710472107),
                ('__label__S1.M12', 0.22077195346355438),
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
                ('__label__S6.M8', 1.0000090599060059),
                ('__label__S6.M3', 1.056917153618997e-05),
                ('__label__S6.M6', 1.0238982213195413e-05),
            ],
        )
    ],
    ids=["AI & Automation"],
)
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
def test_download(remote, local, expected_exception):
    with expected_exception:
        assert isinstance(Model.download(remote, local), Model)
