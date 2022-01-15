#!/usr/bin/env python

import argparse
from typing import List

from module_classifier.preprocessing.settings import (
    MAIN_EDITION_MERGED_LABEL_FIELD,
    MAIN_EDITION_TEXT_FIELDS,
)
from module_classifier.training.binary_trainer import MainEditionTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Train a new classifier model for the main edition."
    )
    parser.add_argument(
        "--main-edition-file",
        "-m",
        type=argparse.FileType(),
        metavar="FILE",
        required=True,
        help="The file containing the main edition items with `link_id` column.",
    )
    parser.add_argument(
        "--archive-file",
        "-a",
        type=argparse.FileType(),
        metavar="FILE",
        required=True,
        help="The file containing the archive items with input texts.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("wb"),
        required=True,
        metavar="FILE",
        help="The model binary output file.",
    )
    parser.add_argument(
        "--text-fields",
        type=List[str],
        nargs="+",
        metavar="COLUMN",
        default=MAIN_EDITION_TEXT_FIELDS,
        help=f"The column(s) containing the text fields in the input file. Defaults to '{' '.join(MAIN_EDITION_TEXT_FIELDS)}'.",
    )
    parser.add_argument("--quantize", action="store_false", help="Quantize the model.")

    args = parser.parse_args()

    trainer = MainEditionTrainer()

    trainer.train_model(
        args.archive_file.name,
        args.output.name,
        text_fields=args.text_fields,
        class_field=MAIN_EDITION_MERGED_LABEL_FIELD,
        main_edition_file=args.main_edition_file.name,
        quantize=args.quantize,
    )
