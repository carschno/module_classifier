import csv
import logging
import os
from tempfile import NamedTemporaryFile
from typing import Collection, IO, Iterable, Mapping, Optional

import fasttext
from fasttext import FastText
from module_classifier.preprocess import fasttext_line
from module_classifier.settings import CLASS_FIELD, TEXT_FIELDS

from ..settings import QUANTIZE, TRAINING_PARAMS


class Trainer:
    def __init__(
        self, training_parameters: Mapping[str, float] = TRAINING_PARAMS
    ):
        """Initialize module classifier trainer. """
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__params = training_parameters

    def train_model(
        self,
        input_file: str,
        target_file: str,
        text_fields: Iterable[str] = TEXT_FIELDS,
        class_field: str = CLASS_FIELD,
        quantize: bool = QUANTIZE,
    ):
        """Train a module classifier model from a CSV file.

        Args:
            input_file: the CSV file to read the data from.
            target_file: the file in which to store the trained model
            text_fields: the CSV columns from which to read the text data
                (default: ["item_title", "authors", "publication_name",
                "abstract_description", "full_text"])
            class_field: the CSV colum from which to read the module name
                (default: module_id_for_all)
            quantize: if true (default), model is compressed using
                quantization before writing to disk.

        """
        with NamedTemporaryFile("wt") as temp_file:
            self._write_training_file(
                input_file, temp_file, text_fields, class_field
            )
            model: FastText = self._train_model(
                temp_file.name, target_file=None if quantize else target_file
            )
            if quantize:
                self._quantize(model, temp_file.name, target_file)

    def _write_training_file(
        self,
        input_file: str,
        target_file: IO[str],
        text_fields: Iterable[str],
        class_field: str,
    ):
        self.__logger.info(f"Reading input file '{input_file}'...")
        with open(input_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            lines: Iterable[str] = (
                fasttext_line(row, text_fields, class_field)
                for row in reader
                if row.get(class_field)
            )
            self.__logger.info(
                f"Writing temporary training file to '{target_file.name}'..."
            )
            for line in lines:
                target_file.write(line + os.linesep)
        target_file.flush()

    def _train_model(
        self, training_file: str, target_file: Optional[str]
    ) -> FastText:
        self.__logger.info("Training model...")
        model = fasttext.train_supervised(training_file, **self.__params)
        if target_file:
            self.__logger.info(
                f"Writing trained model to file '{target_file}'..."
            )
            model.save_model()
        return model

    def _quantize(self, model: FastText, training_file: str, target_file: str):
        self.__logger.info("Compressing model...")
        model.quantize(training_file, retrain=True)

        self.__logger.info(f"Writing trained model to file '{target_file}'...")
        model.save_model(target_file)