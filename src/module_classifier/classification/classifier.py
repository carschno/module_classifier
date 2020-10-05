import logging
import os
from typing import Dict, List, Tuple

import fasttext

from ..preprocess import clean, fasttext_line
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
        return list(zip(labels, probabilities))

    @classmethod
    def download(cls, url: str, local_path: str):
        if os.path.exists(local_path):
            logging.info(
                f"Local file '{local_path}' already exists, skipping download."
            )
        else:
            # TODO: download remote model
            raise NotImplementedError()
        return cls(local_path)
