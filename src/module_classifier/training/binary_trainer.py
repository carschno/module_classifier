import csv
import os
from typing import IO, Iterable

from ..classification.binary_classifier import BinaryClassifier
from . import Trainer


class BinaryTrainer(Trainer):
    def _write_training_file(
        self,
        input_file: str,
        target_file: IO[str],
        text_fields: Iterable[str],
        class_field: str,
        **kwargs,
    ):
        self.logger.info(f"Reading input file '{input_file}'...")
        self.logger.info(f"Writing temporary FastText file to '{target_file.name}'...")

        with open(input_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if class_field in row:
                    target_file.write(
                        BinaryClassifier.fasttext_line(row, text_fields, class_field)
                    )
                    target_file.write(os.linesep)

        target_file.flush()
