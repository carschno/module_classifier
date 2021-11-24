import string
from pathlib import Path
from typing import Iterable

CWD: Path = Path(__file__).parent

TEXT_FIELDS: Iterable[str] = (
    "item_title",
    "authors",
    "publication_name",
    "abstract_description",
    "excerpts_ts",
    "yt_description",
)

CLASS_FIELD: str = "module_id_for_all"
MIN_TOKEN_LENGTH: int = 3
PUNCTUATION_CHARACTERS: str = string.punctuation + "–‒—‘’”“"

DEFAULT_MODEL: str = str(CWD / "data" / "classifier.model.ftz")

DEFAULT_MODULE_DELIMITER: str = "_"
