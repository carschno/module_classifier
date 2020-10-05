import logging
import os
from typing import Dict, Iterable, List, Tuple

import fasttext

from ..preprocess import clean, fasttext_line
from ..preprocess.preprocessing import LABEL_PREFIX
from ..settings import DEFAULT_MODEL


class Classifier:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model = fasttext.load_model(path=model_path)

    def predict_row(
        self, row: Dict[str, str], k: int = 1
    ) -> List[Tuple[str, float]]:
        return self.predict_text(fasttext_line(row), k)

    def predict_text(self, text: str, k: int = 1) -> List[Tuple[str, float]]:
        labels: List[str]
        probabilities: List[float]
        labels, probabilities = self.model.predict(clean(text), k)
        return Classifier._post_process(zip(labels, probabilities))

    @staticmethod
    def _post_process(predictions: Iterable[Tuple[str, float]]):
        return [
            (label[len(LABEL_PREFIX) :], prob) for label, prob in predictions
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
