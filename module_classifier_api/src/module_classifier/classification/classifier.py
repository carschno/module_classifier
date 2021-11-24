import logging
import os
from typing import Dict, Iterable, List, Tuple

import fasttext

from ..models import Module
from ..preprocess import clean, fasttext_line
from ..preprocess.preprocessing import LABEL_PREFIX, MODULE_DELIMITERS
from ..settings import DEFAULT_MODEL


class Classifier:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model = fasttext.load_model(path=model_path)

    def predict_row(
        self,
        row: Dict[str, str],
        k: int = 1,
        columns: Iterable[str] = (),
    ) -> List[Tuple[Module, float]]:
        """
        Predict label for a CSV row.

        Args:
            row:    a dictionary
            k:  the number of predictions to output;
            columns:    a list of column names to extract text from;
                if not specified (or empty), uses all fields in the row.

        Returns:
                a list of tuples where each tuple contains the predicted label
                plus the model confidence for that label.

        """
        return self.predict_text(fasttext_line(row, columns), k)

    def predict_text(self, text: str, k: int = 1) -> List[Tuple[Module, float]]:
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
