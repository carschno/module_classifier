import os
from tempfile import TemporaryDirectory

import pytest
from src.module_classifier.classification import Classifier
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


@pytest.mark.parametrize(
    "filename,content,expected,exception",
    [
        ("test_file", b"", None, pytest.raises(ValueError)),
        ("test_file.00000000000000000000000000000000", b"", False, does_not_raise()),
        ("test_file.d41d8cd98f00b204e9800998ecf8427e", b"", True, does_not_raise()),
        (
            "test_file.d41d8cd98f00b204e9800998ecf8427e",
            b"some text",
            False,
            does_not_raise(),
        ),
        (
            "test_file.552e21cd4cd9918678e3c1a0df491bc3",
            b"some text",
            True,
            does_not_raise(),
        ),
    ],
)
def test_validate_md5(filename, content, expected, exception):
    with TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, filename)

        with open(filename, "wb") as test_file:
            test_file.write(content)
            test_file.flush()
            with exception:
                assert Classifier.validate_md5(test_file.name) == expected
