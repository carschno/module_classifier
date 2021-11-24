#!/usr/bin/env python

import argparse

from module_classifier.settings import DEFAULT_MODULE_DELIMITER
from module_trainer.training import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a model against a file.")
    parser.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("r"),
        required=True,
        help="The input CSV file.",
    )

    parser.add_argument(
        "--model-file",
        "-m",
        type=argparse.FileType("rb"),
        required=True,
        help="The model file.",
    )

    parser.add_argument(
        "--delimiter",
        "-d",
        type=str,
        required=False,
        help=f"The delimiter used in module labels as in 'S1.M1' or 'S1_M1'. If not given, uses default ('{DEFAULT_MODULE_DELIMITER}').",
    )

    parser.add_argument(
        "--labels", "-l", action="store_true", help="Print model labels only."
    )

    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("--threshold", "-t", type=float, default=0.0)

    args = parser.parse_args()

    if args.labels:
        from fasttext import FastText

        model = FastText.load_model(args.model_file.name)
        labels, frequencies = model.get_labels(include_freq=True)
        for l, f in zip(labels, frequencies):
            print(f"{l}\t{f}")

    else:
        trainer = Trainer()
        n, precision, recall = trainer.evaluate_model(
            args.input.name,
            args.model_file.name,
            k=args.k,
            threshold=args.threshold,
            module_delimiter=args.delimiter,
        )

        print(f"Number of samples:\t{n}")
        print(f"Precision:\t{precision}")
        print(f"Recall:\t{recall}")
