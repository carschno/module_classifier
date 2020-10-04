from typing import Dict, List, Tuple

import fasttext

from ..preprocess import fasttext_line


class Model:
    def __init__(self, model_path: str):
        self.model = fasttext.load_model(path=model_path)

    def predict(
        self, row: Dict[str, str], k: int = 1
    ) -> List[Tuple[str, float]]:
        text = fasttext_line(row)

        labels: List[str]
        probabilities: List[float]
        labels, probabilities = self.model.predict(text, k)

        return list(zip(labels, probabilities))
