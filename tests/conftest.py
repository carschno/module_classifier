from contextlib import contextmanager
from pathlib import Path

CWD: Path = Path(__file__).parent

TEST_MODEL: str = str(CWD / "data" / "model" / "fasttext.txt.model.bin")

@contextmanager
def does_not_raise():
    yield
