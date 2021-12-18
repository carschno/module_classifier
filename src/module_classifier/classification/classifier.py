from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import numpy as np

from ..preprocessing.settings import LABEL_PREFIX


class Classifier(ABC):
    @property
    def raw_labels(self) -> List[str]:
        return self.model.get_labels()

    @property
    def labels(self) -> List[str]:
        return [label[len(LABEL_PREFIX) :] for label in self.raw_labels]

    @abstractmethod
    def predict_texts(self, texts: List[str], k: int = 1) -> List[Any]:
        return NotImplemented

    def predict_text(self, text: str, k: int = 1) -> List[Any]:
        return self.predict_texts([text], k)[0]

    def predict_rows(
        self, rows: List[Dict[str, str]], k: int = 1, columns: Iterable[str] = ()
    ) -> List[Any]:
        return NotImplemented

    def prediction_probs(self, texts: List[str], k: int) -> np.ndarray:
        predictions: List[Any] = self.predict_texts(texts, k)
        all_probs: List[np.ndarray] = [
            p.get_probabilities(self.raw_labels) for p in predictions
        ]
        return np.array(all_probs)
