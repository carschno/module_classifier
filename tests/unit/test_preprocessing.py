import pytest

from src.module_classifier.preprocess.preprocessing import clean, fasttext_line
from tests.conftest import does_not_raise


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", ""),
        ("random tokens", "random tokens"),
        ("CAPITALS", "capitals"),
        ("Capitals", "capitals"),
        ("short to", "short"),
        ("spe-cial character", "spe cial character"),
        ("extra  whitespace", "extra whitespace"),
        ("no  extra & whitespace", "extra whitespace"),
    ],
    ids=[
        "empty string",
        "standard string",
        "all CAPS",
        "some caps",
        "short token",
        "special character",
        "extra whitespace",
        "extra whitespace after removing short token",
    ],
)
def test_clean(text, expected):
    assert clean(text) == expected


@pytest.mark.parametrize(
    "row,expected,exception",
    [
        ({}, None, pytest.raises(ValueError)),
        ({"title": ""}, None, pytest.raises(ValueError)),
        (
            {
                "item_title": "",
                "authors": "",
                "publication_name": "",
                "abstract_description": "",
            },
            "__label__    ",
            does_not_raise(),
        ),
        (
            {
                "item_title": "test title",
                "authors": "test author",
                "publication_name": "test publication",
                "abstract_description": "test abstract",
            },
            "__label__ test title test author test publication test abstract",
            does_not_raise(),
        ),
    ],
    ids=["empty row", "missing fields", "empty fields", "all fields"],
)
def test_fasttext_line(row, expected, exception):
    with exception:
        assert fasttext_line(row) == expected
