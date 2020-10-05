from contextlib import contextmanager
from pathlib import Path
from typing import Optional

CWD: Path = Path(__file__).parent

TEST_MODEL: Optional[str] = None


@contextmanager
def does_not_raise():
    yield
