import string
from typing import Iterable

TEXT_FIELDS: Iterable[str] = (
    "item_title",
    "authors",
    "publication_name",
    "abstract_description",
)
CLASS_FIELD: str = "module_id_for_all"
MIN_TOKEN_LENGTH: int = 3
PUNCTUATION_CHARACTERS: str = string.punctuation + "–‒—‘’”“"
