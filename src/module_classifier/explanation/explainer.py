from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from lime.explanation import Explanation
from lime.lime_text import LimeTextExplainer

from ..classification import Classifier
from ..preprocessing import clean


class Explainer:
    def __init__(self, classifier: Classifier) -> None:
        self._classifier: Classifier = classifier
        self._explainer: LimeTextExplainer = LimeTextExplainer(
            split_expression=lambda x: clean(x).split(),
            bow=False,
            class_names=classifier.raw_labels,
        )

    def explain(self, input: str, k: int = 1, **kwargs) -> Explanation:
        return self._explainer.explain_instance(
            clean(input),
            classifier_fn=lambda x: self._classifier.prediction_probs(x, k=k),
            top_labels=k,
            **kwargs
        )
