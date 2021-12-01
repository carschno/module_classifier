from pathlib import Path

CWD: Path = Path(__file__).parent

DEFAULT_MODEL: str = str(CWD.parent / "data" / "classifier.model.ftz")
