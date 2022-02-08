from distutils.util import strtobool
from typing import Dict, Iterable, List, Tuple

from ..preprocessing.settings import (
    LABEL_PREFIX,
    MAIN_EDITION_MERGED_LABEL_FIELD,
    MAIN_EDITION_TEXT_FIELDS,
)
from .classifier import Classifier
from .settings import (
    AWS_S3_MODELS_BUCKET,
    MAIN_EDITION_CLASSIFIER_MODEL_FILE_NAME,
    MAIN_EDITION_CLASSIFIER_MODEL_PATH,
)


class BinaryClassifier(Classifier):
    @staticmethod
    def _deserialize_label(label: str) -> bool:
        return bool(strtobool(label[len(LABEL_PREFIX) :]))


class MainEditionClassifier(BinaryClassifier):
    def predict_texts(self, texts: List[str], k: int = 1) -> List[Tuple[bool, float]]:
        return self._predict(texts, k)

    def _predict(self, texts: List[str], k: int) -> List[Tuple[bool, float]]:
        labels, probs = self.model.predict(texts, k)
        return [
            (self._deserialize_label(label[0]), float(prob))
            for label, prob in zip(labels, probs)
        ]

    @staticmethod
    def fasttext_line(
        row: Dict[str, str],
        text_fields: Iterable[str] = MAIN_EDITION_TEXT_FIELDS,
        class_field: str = MAIN_EDITION_MERGED_LABEL_FIELD,
        **kwargs
    ) -> str:

        # TODO: remove this method (no need to override)
        return Classifier.fasttext_line(row, text_fields, class_field, **kwargs)

    @classmethod
    def from_s3(
        cls,
        bucket: str = AWS_S3_MODELS_BUCKET,
        object_name: str = MAIN_EDITION_CLASSIFIER_MODEL_FILE_NAME,
        local_path: str = MAIN_EDITION_CLASSIFIER_MODEL_PATH,
        check_md5: bool = True,
    ) -> Classifier:

        return super().from_s3(bucket, object_name, local_path, check_md5)
