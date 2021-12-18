from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import numpy as np
from fasttext.FastText import _FastText

from ..preprocessing.settings import CLASS_FIELD, LABEL_PREFIX, TEXT_FIELDS


class Classifier(ABC):
    model: _FastText

    @property
    def raw_labels(self) -> List[str]:
        return self.model.get_labels()

    @property
    def labels(self) -> List[str]:
        return [label[len(LABEL_PREFIX) :] for label in self.raw_labels]

    @abstractmethod
    def predict_texts(self, texts: List[str], k: int = 1) -> List[Any]:
        return NotImplemented

    @abstractmethod
    def _predict(self, texts: List[str], k: int) -> List[Any]:
        return NotImplemented

    @staticmethod
    @abstractmethod
    def fasttext_line(
        row: Dict[str, str],
        text_fields: Iterable[str] = TEXT_FIELDS,
        class_field: str = CLASS_FIELD,
        **kwargs
    ) -> str:
        return NotImplemented

    def predict_text(self, text: str, k: int = 1) -> List[Any]:
        return self.predict_texts([text], k)[0]

    def predict_rows(
        self, rows: List[Dict[str, str]], k: int = 1, columns: Iterable[str] = ()
    ) -> List[Any]:
        return self._predict([self.fasttext_line(row, columns) for row in rows], k)

    def prediction_probs(self, texts: List[str], k: int) -> np.ndarray:
        predictions: List[Any] = self.predict_texts(texts, k)
        all_probs: List[np.ndarray] = [
            p.get_probabilities(self.raw_labels) for p in predictions
        ]
        return np.array(all_probs)
