from contextlib import contextmanager
import os
from pathlib import Path
from typing import Optional

from src.module_classifier.classification.settings import (
    MODULE_CLASSIFIER_DEFAULT_MODEL_PATH,
)

CWD: Path = Path(__file__).parent

TEST_MODEL: Optional[str] = (
    MODULE_CLASSIFIER_DEFAULT_MODEL_PATH
    if os.path.isfile(MODULE_CLASSIFIER_DEFAULT_MODEL_PATH)
    else None
)
TEST_ARCHIVE_FILE: Path = CWD / "data" / "test_archive_items.csv"
TEST_MAIN_EDITION_ITEMS_FILE: Path = CWD / "data" / "test_main_edition_items.csv"


@contextmanager
def does_not_raise():
    yield
