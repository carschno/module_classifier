import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import boto3
import fasttext
import numpy as np
from fasttext.FastText import _FastText

from ..preprocessing import clean
from ..preprocessing.settings import CLASS_FIELD, LABEL_PREFIX, TEXT_FIELDS


class Classifier(ABC):
    model: _FastText

    def __init__(self, model_path: str):
        self.model = fasttext.load_model(path=model_path)

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
    def fasttext_line(
        row: Dict[str, str],
        text_fields: Iterable[str] = TEXT_FIELDS,
        class_field: str = CLASS_FIELD,
        **kwargs,
    ) -> str:
        if text_fields:
            # validate that specified text fields are present
            for field in text_fields:
                if field not in row:
                    raise ValueError(f"Missing input field: '{field}'.")
        else:
            text_fields = row.keys()

        label: str = LABEL_PREFIX + row[class_field] if class_field in row else ""
        return " ".join([label] + [clean(row[field]) for field in text_fields]).strip()

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

    @classmethod
    @abstractmethod
    def from_s3(cls, bucket: str, object_name: str, local_path: str):
        if os.path.exists(local_path):
            logging.info(
                f"Local file '{local_path}' already exists, skipping download."
            )
        else:
            logging.info(
                f"Downloading model from 's3://{bucket}/{object_name}' to '{local_path}'."
            )
            s3 = boto3.client("s3")
            s3.download_file(bucket, object_name, local_path)

        return cls(local_path)
