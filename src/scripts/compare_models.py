import argparse
from typing import Iterable, List, Set, Tuple

from fasttext import FastText

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare two models.")
    parser.add_argument(
        "--models",
        nargs=2,
        required=True,
        type=argparse.FileType("rb"),
        help="The two model files.",
    )

    args = parser.parse_args()

    models: List = [FastText.load_model(f.name) for f in args.models]

    model_labels: List[Tuple[List[str], Iterable[int]]] = [
        model.get_labels(include_freq=True) for model in models
    ]

    common_labels: Set[str] = set(model_labels[0][0]).intersection(model_labels[1][0])
    extra_labels_1: Set[str] = set(model_labels[0][0]).difference(model_labels[1][0])
    extra_labels_2: Set[str] = set(model_labels[1][0]).difference(model_labels[0][0])

    print(f"Common labels:\t{len(common_labels)}")
    print(f"Extra labels 1:\t{sorted(list(extra_labels_1))}")
    print(f"Extra labels 2:\t{sorted(list(extra_labels_2))}")
