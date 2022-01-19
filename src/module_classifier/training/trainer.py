import logging
import os
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
from typing import IO, Any, Dict, Iterable, Optional

import fasttext
from fasttext import FastText

from ..preprocessing.settings import CLASS_FIELD, TEXT_FIELDS
from .settings import QUANTIZE, TRAINING_PARAMS


class Trainer(ABC):
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
        **kwargs,
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
                input_file, temp_file, text_fields, class_field, **kwargs
            )
            if quantize:
                model: FastText = self._train_model(temp_file.name, None)
                self._quantize(model, temp_file.name, target_file)
            else:
                self._train_model(temp_file.name, target_file)

    @abstractmethod
    def evaluate_model(
        self,
        input_file: str,
        model_file: str,
        module_delimiter: Optional[str] = None,
        test_label: bool = False,
        **kwargs,
    ):
        return NotImplemented

    @abstractmethod
    def _write_training_file(
        self,
        input_file: str,
        target_file: IO[str],
        text_fields: Iterable[str],
        class_field: str,
        **kwargs,
    ):
        return NotImplemented

    def _train_model(
        self,
        training_file: str,
        target_file: Optional[str],
        training_params: Dict[str, Any] = TRAINING_PARAMS,
    ) -> FastText:
        self.__logger.info(f"Training model. Arguments: {training_params}")
        model = fasttext.train_supervised(training_file, **training_params)
        if target_file:
            self.__logger.info(f"Writing trained model to file '{target_file}'...")
            model.save_model(target_file)
        return model

    def _quantize(self, model: FastText, training_file: str, target_file: str):
        self.__logger.info("Compressing model...")
        model.quantize(training_file, retrain=True)

        self.__logger.info(f"Writing trained model to file '{target_file}'...")
        model.save_model(target_file)
