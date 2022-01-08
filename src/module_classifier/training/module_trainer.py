import csv
import os
from tempfile import NamedTemporaryFile
from typing import IO, Iterable

from fasttext import FastText

from ..classification import ModuleClassifier
from ..preprocessing.settings import CLASS_FIELD, DEFAULT_MODULE_DELIMITER, TEXT_FIELDS
from . import Trainer


class ModuleTrainer(Trainer):
    def evaluate_model(
        self,
        input_file: str,
        model_file: str,
        module_delimiter: str = DEFAULT_MODULE_DELIMITER,
        test_label: bool = False,
        **kwargs,
    ):
        with NamedTemporaryFile("wt") as temp_file:
            self._write_training_file(
                input_file,
                temp_file,
                TEXT_FIELDS,
                CLASS_FIELD,
                module_delimiter=module_delimiter,
            )

            self.logger.info("Loading model from '%s'", model_file)
            model: FastText._FastText = FastText.load_model(model_file)

            return (
                model.test_label(temp_file.name, **kwargs)
                if test_label
                else model.test(temp_file.name, **kwargs)
            )

    def _write_training_file(
        self,
        input_file: str,
        target_file: IO[str],
        text_fields: Iterable[str],
        class_field: str,
        module_delimiter: str = DEFAULT_MODULE_DELIMITER,
    ):
        self.logger.info(f"Reading input file '{input_file}'...")
        self.logger.info(f"Writing temporary FastText file to '{target_file.name}'...")
        with open(input_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if class_field in row:
                    target_file.write(
                        ModuleClassifier.fasttext_line(
                            row,
                            text_fields,
                            class_field,
                            module_delimiter=module_delimiter,
                        )
                    )
                    target_file.write(os.linesep)

        target_file.flush()
