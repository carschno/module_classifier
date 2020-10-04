import os
from typing import Dict

from ..settings import (
    CLASS_FIELD,
    MIN_TOKEN_LENGTH,
    PUNCTUATION_CHARACTERS,
    TEXT_FIELDS,
)


def clean(s: str) -> str:
    for c in PUNCTUATION_CHARACTERS:
        s = s.replace(c, " ")

    # Replace all numbers with 0
    for n in "0123456789":
        s = s.replace(n, "0")

    # Remove linebreaks, lowercase, remove short tokens
    return " ".join(
        (
            token
            for token in s.strip().replace(os.linesep, " ").lower().split()
            if len(token) >= MIN_TOKEN_LENGTH
        )
    )


def fasttext_line(row: Dict[str, str]) -> str:
    # TODO: make row a TypedDict
    for field in TEXT_FIELDS:
        if field not in row:
            raise ValueError(f"Missing input field: '{field}'.")

    return " ".join(
        [f"__label__{row.get(CLASS_FIELD, '').upper()}"]
        + [clean(row[key]) for key in TEXT_FIELDS]
    )
