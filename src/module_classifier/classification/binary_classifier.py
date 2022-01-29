from typing import Dict, Iterable, List, Tuple

from ..preprocessing.settings import (
    MAIN_EDITION_MERGED_LABEL_FIELD,
    MAIN_EDITION_TEXT_FIELDS,
)
from .classifier import Classifier


class BinaryClassifier(Classifier):
    pass


class MainEditionClassifier(BinaryClassifier):
    def predict_texts(self, texts: List[str], k: int = 1) -> List[Tuple[str, float]]:
        return self._predict(texts, k)

    def _predict(self, texts: List[str], k: int) -> List[Tuple[str, float]]:
        labels, probs = self.model.predict(texts, k)
        return [(label, float(prob)) for label, prob in zip(labels, probs)]

    @staticmethod
    def fasttext_line(
        row: Dict[str, str],
        text_fields: Iterable[str] = MAIN_EDITION_TEXT_FIELDS,
        class_field: str = MAIN_EDITION_MERGED_LABEL_FIELD,
        **kwargs
    ) -> str:

        # TODO: remove this method (no need to override)
        return Classifier.fasttext_line(row, text_fields, class_field, **kwargs)
