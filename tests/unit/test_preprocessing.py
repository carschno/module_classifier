import pytest
from src.module_classifier.preprocessing import clean



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

