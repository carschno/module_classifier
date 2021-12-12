import logging
import os
import urllib.request
from dataclasses import dataclass
from hashlib import md5
from posixpath import splitext
from typing import Dict, Iterable, List

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


class Classifier:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model = fasttext.load_model(path=model_path)

    @property
    def raw_labels(self) -> List[str]:
        return self.model.get_labels()

    @property
    def labels(self) -> List[str]:
        return [label[len(LABEL_PREFIX) :] for label in self.raw_labels]

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

    # TODO: return List[List[Prediction]]?
    def predict_rows(
        self, rows: List[Dict[str, str]], k: int = 1, columns: Iterable[str] = ()
    ) -> List[Predictions]:
        return self._predict([fasttext_line(row, columns) for row in rows], k)

    def predict_text(self, text: str, k: int = 1) -> List[Prediction]:
        return self.predict_texts([text], k)[0].to_predictions()

    # TODO: return List[List[Prediction]]?
    def predict_texts(self, texts: List[str], k: int = 1) -> List[Predictions]:
        if not texts:
            raise ValueError("No input text provided.")
        return self._predict([clean(text) for text in texts], k)

    def prediction_probs(self, texts: List[str], k: int) -> np.ndarray:
        predictions: List[Predictions] = self.predict_texts(texts, k)
        all_probs: List[np.ndarray] = [
            p.get_probabilities(self.raw_labels) for p in predictions
        ]
        return np.array(all_probs)

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

    @classmethod
    def download(cls, url: str, local_path: str = DEFAULT_MODEL):
        if os.path.exists(local_path):
            logging.info(
                f"Local file '{local_path}' already exists, skipping download."
            )
        else:
            logging.info(f"Downloading model from '{url}' to '{local_path}'.")
            with open(local_path, "wb") as f, urllib.request.urlopen(url) as response:
                f.write(response.read())

        expected_hash: str = splitext(local_path)[1][1:]
        if len(expected_hash) == 32:
            if Classifier.get_hash(local_path) != expected_hash:
                raise ValueError(
                    f"Expected hash '{expected_hash}' does not match dowloaded model file."
                )
        else:
            logging.warning(
                f"Model file name should end with MD5 sum; invalid hash: '{expected_hash}'"
            )
        return cls(local_path)

    @staticmethod
    def get_hash(filename: str) -> str:
        with open(filename, "rb") as f:
            new_hash: str = md5(f.read()).hexdigest()
        return new_hash
