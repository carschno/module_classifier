import logging
import os
from typing import Dict, Iterable, List, Tuple

import fasttext

from ..preprocess import clean, fasttext_line
from ..preprocess.preprocessing import LABEL_PREFIX
from ..settings import DEFAULT_MODEL, TEXT_FIELDS


class Classifier:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model = fasttext.load_model(path=model_path)

    def predict_row(
        self,
        row: Dict[str, str],
        k: int = 1,
        columns: Iterable[str] = TEXT_FIELDS,
    ) -> List[Tuple[str, float]]:
        """
        Predict label for a CSV row.

        Args:
            row:    a dictionary
            k:  the number of predictions to output;
                defaults to TEXT_FIELDS as specified in the settings.
            columns:    a list of column names to extract text from

        Returns:
                a list of tuples where each tuple contains the predicted label
                plus the model confidence for that label.

        """
        return self.predict_text(fasttext_line(row, columns), k)

    def predict_text(self, text: str, k: int = 1) -> List[Tuple[str, float]]:
        """
        Predict label for any text
        Args:
            text: a text
            k:  the number of predictions to output

        Returns:
                a list of tuples where each tuple contains the predicted label
                plus the model confidence for that label.

        """
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
