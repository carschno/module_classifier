import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import fasttext
import numpy as np

from ..preprocessing import Module, clean, fasttext_line
from ..preprocessing.settings import LABEL_PREFIX, MODULE_DELIMITERS
from .settings import DEFAULT_MODEL


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
    probs: Iterable[float]

    def to_predictions(self) -> List[Prediction]:
        return [
            Prediction.from_label(label, prob)
            for label, prob in zip(self.labels, self.probs)
        ]

    @staticmethod
    def from_fasttext_predictions(
        labels: List[List[str]], probs: List[np.ndarray]
    ) -> List["Predictions"]:
        return [Predictions(_labels, _probs) for _labels, _probs in zip(labels, probs)]


class Classifier:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model = fasttext.load_model(path=model_path)

    @property
    def labels(self) -> List[str]:
        return [label[len(LABEL_PREFIX) :] for label in self.model.get_labels()]

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
        return self._predict([fasttext_line(row, columns)], k)[0].to_predictions()

    def predict_rows(
        self, rows: List[Dict[str, str]], k: int = 1, columns: Iterable[str] = ()
    ) -> List[Predictions]:
        return self._predict([fasttext_line(row, columns) for row in rows], k)

    def predict_text(self, text: str, k: int = 1) -> List[Prediction]:
        return self.predict_texts([text], k)[0].to_predictions()

    def predict_texts(self, texts: List[str], k: int = 1) -> List[Predictions]:
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
    def _post_process(predictions: Iterable[Tuple[str, float]]):

        return [
            (
                Module.from_string(
                    label, label_prefix=LABEL_PREFIX, delimiters=MODULE_DELIMITERS
                ),
                prob,
            )
            for label, prob in predictions
        ]

    @classmethod
    def download(cls, url: str, local_path: str = DEFAULT_MODEL):
        if os.path.exists(local_path):
            logging.info(
                f"Local file '{local_path}' already exists, skipping download."
            )
        else:
            # TODO: download remote model
            raise NotImplementedError()
        return cls(local_path)
