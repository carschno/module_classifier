#!/usr/bin/env python

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict

from module_classifier.preprocessing.settings import DEFAULT_MODULE_DELIMITER
from module_classifier.training.binary_trainer import MainEditionTrainer, Trainer


class Stats(TypedDict):
    precision: Optional[float]
    recall: Optional[float]
    f1score: Optional[float]


@dataclass
class LabelOutput:
    label: str
    stats: Stats

    def to_row(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "precision": self.stats["precision"],
            "recall": self.stats["recall"],
            "f1score": self.stats["f1score"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a model against a file.")
    parser.add_argument(
        "--main-edition-file",
        type=argparse.FileType(),
        metavar="FILE",
        required=True,
        help="The file containing the main edition items with `link_id` column.",
    )
    parser.add_argument(
        "--archive-file",
        type=argparse.FileType(),
        metavar="FILE",
        required=True,
        help="The file containing the archive items with input texts.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The output file.",
    )

    parser.add_argument(
        "--model-file",
        "-m",
        type=argparse.FileType("rb"),
        required=True,
        help="The model file.",
    )

    parser.add_argument(
        "--labels", "-l", action="store_true", help="Run analysis per label."
    )

    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("--threshold", "-t", type=float, default=0.0)

    args = parser.parse_args()

    if args.labels and args.output == sys.stdout:
        raise ValueError("Cannot output per label to stdout.")

    trainer: Trainer = MainEditionTrainer()

    evaluation_args = {
        "input_file": args.archive_file.name,
        "model_file": args.model_file.name,
        "k": args.k,
        "threshold": args.threshold,
        "main_edition_file": args.main_edition_file.name,
    }

    if args.labels:
        print(
            f"Running per label analysis for model '{args.model_file.name}' on data '{args.input.name}'."
        )
        result = trainer.evaluate_model(**evaluation_args, test_label=True)

        writer = csv.DictWriter(
            args.output, fieldnames=["label", "precision", "recall", "f1score"]
        )
        writer.writeheader()
        for label, stats in result.items():
            writer.writerow(LabelOutput(label=label, stats=Stats(**stats)).to_row())

    print(
        f"Running evaluation for model '{args.model_file.name}' on data '{args.archive_file.name}'."
    )
    n, precision, recall = trainer.evaluate_model(**evaluation_args)

    print(f"Number of samples:\t{n}")
    print(f"Precision:\t{precision}")
    print(f"Recall:\t{recall}")
