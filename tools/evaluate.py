#!/usr/bin/env python

import argparse

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

    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("--threshold", "-t", type=float, default=0.0)

    args = parser.parse_args()

    trainer = Trainer()
    n, precision, recall = trainer.evaluate_model(
        args.input.name, args.model_file.name, k=args.k, threshold=args.threshold
    )

    print(f"Number of samples:\t{n}")
    print(f"Precision:\t{precision}")
    print(f"Recall:\t{recall}")
