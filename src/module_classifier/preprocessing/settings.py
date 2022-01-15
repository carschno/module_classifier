import string
from pathlib import Path
from typing import Iterable, Literal

CWD: Path = Path(__file__).parent

TEXT_FIELDS: Iterable[str] = (
    "item_title",
    "authors",
    "publication_name",
    "abstract_description",
    "excerpts_ts",
    "yt_description",
)

MAIN_EDITION_TEXT_FIELDS: Iterable[str] = (
    "publication_name",
    "item_title",
    "excerpt_ts",
)
MAIN_EDITION_MERGED_LABEL_FIELD: str = "label"
MAIN_EDITION_ID_FIELD: str = "link_id"


CLASS_FIELD: str = "module_id_for_all"
MIN_TOKEN_LENGTH: int = 3
PUNCTUATION_CHARACTERS: str = string.punctuation + "–‒—‘’”“"

DEFAULT_MODEL: str = str(CWD / "data" / "classifier.model.ftz")

DEFAULT_MODULE_DELIMITER: str = "_"

LABEL_PREFIX: Literal["__label__"] = "__label__"
MODULE_DELIMITERS: Iterable[str] = ("_", ".")


assert (
    DEFAULT_MODULE_DELIMITER in MODULE_DELIMITERS
), f"Module delimiters must contain default delimiter ('{DEFAULT_MODULE_DELIMITER}')."
