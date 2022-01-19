from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from ..preprocessing import Module, clean
from ..preprocessing.settings import (
    CLASS_FIELD,
    DEFAULT_MODULE_DELIMITER,
    LABEL_PREFIX,
    MODULE_DELIMITERS,
    TEXT_FIELDS,
)
from .classifier import Classifier
from .settings import MODULE_CLASSIFIER_DEFAULT_MODEL


@dataclass
class Prediction:
    """Individual predictions"""

    module: Module
    prob: float

    @classmethod
    def from_label(cls, label: str, prob: float) -> "Prediction":
        return cls(Module.from_string(label, label_prefix=LABEL_PREFIX), prob)


@dataclass
class Predictions:
    """k predictions for an individual input."""

    labels: List[str]
    probs: np.ndarray

    def to_predictions(self) -> List[Prediction]:
        return [
            Prediction.from_label(label, prob)
            for label, prob in zip(self.labels, self.probs)
        ]

    def get_probabilities(self, labels: List[str]) -> np.ndarray:
        probs: List[float] = []
        for label in labels:
            probs.append(
                self.probs[self.labels.index(label)] if label in self.labels else 0.0
            )
        return np.array(probs)

    @staticmethod
    def from_fasttext_predictions(
        labels: List[List[str]], probs: List[np.ndarray]
    ) -> List["Predictions"]:
        return [Predictions(_labels, _probs) for _labels, _probs in zip(labels, probs)]


class ModuleClassifier(Classifier):
    def __init__(self, model_path: str = MODULE_CLASSIFIER_DEFAULT_MODEL):
        return super().__init__(model_path)

    @property
    def modules(self) -> List[Module]:
        return [Module.from_string(label) for label in self.labels]

    def predict_row(
        self,
        row: Dict[str, str],
        k: int = 1,
        columns: Iterable[str] = (),
    ) -> List[Prediction]:
        """
        Predict label for a CSV row.

        Args:
            row:    a dictionary
            k:  the number of predictions to output;
            columns:    a list of column names to extract text from;
                if not specified (or empty), uses all fields in the row.

        Returns:
                a list of Prediction objects of length k,
                where each tuple contains the predicted label
                plus the model confidence for that label.

        """
        return self._predict([self.fasttext_line(row, columns)], k)[0].to_predictions()

    def predict_text(self, text: str, k: int = 1) -> List[Prediction]:
        return self.predict_texts([text], k)[0].to_predictions()

    # TODO: return List[List[Prediction]]?
    def predict_texts(self, texts: List[str], k: int = 1) -> List[Predictions]:
        if not texts:
            raise ValueError("No input text provided.")
        return self._predict([clean(text) for text in texts], k)

    def _predict(self, texts: List[str], k: int) -> List[Predictions]:
        """Predict labels and probabilities for a list of texts.

        Args:
            texts: List[str] Input texts are previsouly pre-processed (cleaned) strings.

        Returns:
           a list of Predictions objects, one per input text.
        """
        if not isinstance(texts, list):
            raise ValueError(f"Invalid input type: {type(texts)}.")

        labels: List[List[str]]
        probs: List[np.ndarray]
        labels, probs = self.model.predict(texts, k)
        return Predictions.from_fasttext_predictions(labels, probs)

    @staticmethod
    def fasttext_line(
        row: Dict[str, str],
        text_fields: Iterable[str] = TEXT_FIELDS,
        class_field: str = CLASS_FIELD,
        *,
        module_delimiter: str = DEFAULT_MODULE_DELIMITER,
    ) -> str:

        if class_field in row:
            module: Module = Module.from_string(
                row[class_field], delimiters=MODULE_DELIMITERS
            )
            row[class_field] = module.to_string(delimiter=module_delimiter)

        return Classifier.fasttext_line(row, text_fields, class_field)

    @classmethod
    def from_s3(
        cls,
        bucket: str,
        object_name: str,
        local_path: str = MODULE_CLASSIFIER_DEFAULT_MODEL,
    ):
        return super().from_s3(bucket, object_name, local_path)
