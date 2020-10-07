import os
from typing import Dict, Iterable, Literal

from ..settings import (
    CLASS_FIELD,
    MIN_TOKEN_LENGTH,
    PUNCTUATION_CHARACTERS,
    TEXT_FIELDS,
)

LABEL_PREFIX: Literal["__label__"] = "__label__"


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


def fasttext_line(
    row: Dict[str, str],
    text_fields: Iterable[str] = TEXT_FIELDS,
    class_field: str = CLASS_FIELD,
) -> str:
    if text_fields:
        # validate that specified text fields are present
        for field in text_fields:
            if field not in row:
                raise ValueError(f"Missing input field: '{field}'.")
    else:
        text_fields = row.keys()

    return " ".join(
        [f"{LABEL_PREFIX}{row.get(class_field, '').upper()}"]
        + [clean(row[key]) for key in text_fields]
    )
