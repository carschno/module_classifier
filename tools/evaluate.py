#!/usr/bin/env python

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict

from module_classifier.settings import DEFAULT_MODULE_DELIMITER
from module_trainer.training import Trainer


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
        "--input",
        "-i",
        type=argparse.FileType("r"),
        required=True,
        help="The input CSV file.",
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
        "--delimiter",
        "-d",
        type=str,
        required=False,
        help=f"The delimiter used in module labels as in 'S1.M1' or 'S1_M1'. If not given, uses default ('{DEFAULT_MODULE_DELIMITER}').",
    )

    parser.add_argument(
        "--labels", "-l", action="store_true", help="Run analysis per label."
    )

    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("--threshold", "-t", type=float, default=0.0)

    args = parser.parse_args()

    if args.labels and args.output == sys.stdout:
        raise ValueError("Cannot output per label to stdout.")

    trainer: Trainer = Trainer()

    if args.labels:
        result = trainer.evaluate_model(
            args.input.name,
            args.model_file.name,
            k=args.k,
            threshold=args.threshold,
            module_delimiter=args.delimiter,
            test_label=args.labels,
        )

        writer = csv.DictWriter(
            args.output, fieldnames=["label", "precision", "recall", "f1score"]
        )
        writer.writeheader()
        for label, stats in result.items():
            writer.writerow(LabelOutput(label=label, stats=Stats(stats)).to_row())

    n, precision, recall = trainer.evaluate_model(
        args.input.name,
        args.model_file.name,
        k=args.k,
        threshold=args.threshold,
        module_delimiter=args.delimiter,
        test_label=False,
    )
    print(f"Number of samples:\t{n}")
    print(f"Precision:\t{precision}")
    print(f"Recall:\t{recall}")
