from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.module_classifier.classification.settings import DEFAULT_MODEL

CWD: Path = Path(__file__).parent

TEST_MODEL: Optional[str] = DEFAULT_MODEL


@contextmanager
def does_not_raise():
    yield
