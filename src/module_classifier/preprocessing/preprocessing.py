import os
from typing import Dict, Iterable, Optional

from .models import Module
from .settings import (
    CLASS_FIELD,
    DEFAULT_MODULE_DELIMITER,
    LABEL_PREFIX,
    MIN_TOKEN_LENGTH,
    MODULE_DELIMITERS,
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


def fasttext_line(
    row: Dict[str, str],
    text_fields: Iterable[str] = TEXT_FIELDS,
    class_field: str = CLASS_FIELD,
    *,
    module_delimiter: Optional[str] = None,
    module_fields: Optional[Iterable[str]] = None,
) -> str:
    if text_fields:
        # validate that specified text fields are present
        for field in text_fields:
            if field not in row:
                raise ValueError(f"Missing input field: '{field}'.")
    else:
        text_fields = row.keys()

    module: str = (
        Module.from_string(
            row.get(class_field, ""), delimiters=MODULE_DELIMITERS
        ).fasttext(
            label_prefix=LABEL_PREFIX,
            fields=module_fields,
            delimiter=module_delimiter or DEFAULT_MODULE_DELIMITER,
        )
        if class_field in row
        else ""
    )

    return " ".join([module] + [clean(row[key]) for key in text_fields]).strip()
