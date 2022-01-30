from pathlib import Path

CWD: Path = Path(__file__).parent
DATA_DIR: Path = CWD.parent / "data"

MAIN_EDITION_CLASSIFIER_MODEL_MD5: str = "0b93fa05a36164033fffccba4a1c34ce"
MAIN_EDITION_CLASSIFIER_MODEL_FILE_NAME: str = (
    f"main_edition.ftz.{MAIN_EDITION_CLASSIFIER_MODEL_MD5}"
)

MODULE_CLASSIFIER_DEFAULT_MODEL: str = str(DATA_DIR / "classifier.model.ftz")
MAIN_EDITION_CLASSIFIER_MODEL_PATH: str = str(
    DATA_DIR / MAIN_EDITION_CLASSIFIER_MODEL_FILE_NAME
)

AWS_S3_MODELS_BUCKET: str = "ts-shared-models"
