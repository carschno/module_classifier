import pytest
from src.module_classifier.classification import ModuleClassifier
from src.module_classifier.explanation import Explainer

from ..conftest import TEST_MODEL


def test_explain():
    expected = {
        3: [
            ("test", pytest.approx(-0.16820062007910383, abs=0.5)),
            ("text", pytest.approx(0.37238473285178575, abs=0.5)),
        ]
    }
    text = "test text"

    explainer = Explainer(ModuleClassifier(TEST_MODEL))
    explanation = explainer.explain(text, k=1, num_features=10, num_samples=10)

    for label in explanation.top_labels:
        assert sorted(explanation.as_list(label)) == sorted(expected[label])
