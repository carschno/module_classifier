import csv
import logging
import os
from tempfile import NamedTemporaryFile
from typing import Dict, IO, Iterable, Optional

import fasttext
from fasttext import FastText
from module_classifier.preprocess import fasttext_line
from module_classifier.settings import CLASS_FIELD, TEXT_FIELDS

from ..settings import QUANTIZE, TRAINING_PARAMS


class Trainer:
    def __init__(
        self,
        *,
        cpus: Optional[int] = None,
        training_parameters: Dict[str, float] = TRAINING_PARAMS,
    ):
        """Initialize module classifier trainer.

        Args:
            cpus: the number of cpus/threads to use for training;
                defaults to the number of CPUs available on the system.
            training_parameters: training parameters
        """
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.setLevel(logging.INFO)

        self.__params = training_parameters
        if cpus:
            cpu_count = os.cpu_count()
            if cpu_count is not None and cpus > cpu_count:
                raise ValueError(
                    f"Requested number of CPUs ({cpus}) must not exceed "
                    f"number of available CPUs ({cpu_count})."
                )
            else:
                self.__params["thread"] = cpus

    @property
    def logger(self) -> logging.Logger:
        return self.__logger

    def train_model(
        self,
        input_file: str,
        target_file: str,
        *,
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
            self._write_training_file(input_file, temp_file, text_fields, class_field)
            if quantize:
                model: FastText = self._train_model(temp_file.name, None)
                self._quantize(model, temp_file.name, target_file)
            else:
                self._train_model(temp_file.name, target_file)

    def evaluate_model(
        self,
        input_file: str,
        model_file: str,
        module_delimiter: Optional[str] = None,
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
            model = FastText.load_model(model_file)
            return model.test(temp_file.name, **kwargs)

    def _write_training_file(
        self,
        input_file: str,
        target_file: IO[str],
        text_fields: Iterable[str],
        class_field: str,
        module_delimiter: Optional[str] = None,
    ):
        self.__logger.info(f"Reading input file '{input_file}'...")
        with open(input_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            lines: Iterable[str] = (
                fasttext_line(
                    row, text_fields, class_field, module_delimiter=module_delimiter
                )
                for row in reader
                if row.get(class_field)
            )
            self.__logger.info(
                f"Writing temporary FastText file to '{target_file.name}'..."
            )
            for line in lines:
                target_file.write(line + os.linesep)
        target_file.flush()

    def _train_model(self, training_file: str, target_file: Optional[str]) -> FastText:
        self.__logger.info("Training model...")
        model = fasttext.train_supervised(training_file, **self.__params)
        if target_file:
            self.__logger.info(f"Writing trained model to file '{target_file}'...")
            model.save_model(target_file)
        return model

    def _quantize(self, model: FastText, training_file: str, target_file: str):
        self.__logger.info("Compressing model...")
        model.quantize(training_file, retrain=True)

        self.__logger.info(f"Writing trained model to file '{target_file}'...")
        model.save_model(target_file)
