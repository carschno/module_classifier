import pytest
from src.module_classifier.classification.module_classifier import ModuleClassifier

from ..conftest import does_not_raise


def test_init():
    with pytest.raises(ValueError):
        ModuleClassifier("/does/not/exist")


@pytest.mark.parametrize(
    "row,columns,expected,exception",
    [
        (
            {},
            (
                "item_title",
                "authors",
                "publication_name",
                "abstract_description",
            ),
            None,
            pytest.raises(ValueError),
        ),
        (
            {"title": ""},
            (
                "item_title",
                "authors",
                "publication_name",
                "abstract_description",
            ),
            None,
            pytest.raises(ValueError),
        ),
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
            "",
            does_not_raise(),
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
            "test title test author test publication test abstract",
            does_not_raise(),
        ),
    ],
    ids=["empty row", "missing fields", "empty fields", "all fields"],
)
def test_fasttext_line(row, columns, expected, exception):
    with exception:
        assert ModuleClassifier.fasttext_line(row, columns) == expected
