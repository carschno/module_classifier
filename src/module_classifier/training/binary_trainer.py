import csv
import os
from tempfile import NamedTemporaryFile
from typing import IO, Iterable, Optional

from fasttext import FastText

from ..classification.binary_classifier import BinaryClassifier
from ..preprocessing.archive_files import ArchiveFile, MainEditionFile, merge_data
from ..preprocessing.settings import (
    MAIN_EDITION_MERGED_LABEL_FIELD,
    MAIN_EDITION_TEXT_FIELDS,
)
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

    def evaluate_model(
        self,
        input_file: str,
        model_file: str,
        module_delimiter: Optional[str] = None,
        test_label: bool = False,
        **kwargs,
    ):
        return NotImplemented


class MainEditionTrainer(BinaryTrainer):
    # TODO: sync signature with abstract Trainer._write_training_file
    def _write_training_file(
        self,
        input_file: str,  # archive_file
        target_file: IO[str],
        text_fields: Iterable[str],
        class_field: str,
        main_edition_file: str,
        **kwargs,
    ):
        archive = ArchiveFile(open(input_file))
        main_edition = MainEditionFile(open(main_edition_file))

        with NamedTemporaryFile("w", delete=False) as merged_file:
            merge_data(archive, main_edition, merged_file, class_field)

        super()._write_training_file(
            merged_file.name, target_file, text_fields, class_field, **kwargs
        )

        os.remove(merged_file.name)

    def evaluate_model(
        self,
        input_file: str,
        model_file: str,
        *,
        main_edition_file: str,
        class_field: str = MAIN_EDITION_MERGED_LABEL_FIELD,
        test_label: bool = False,
        **kwargs,
    ):
        with NamedTemporaryFile("wt", delete=False) as target_file:
            self._write_training_file(
                input_file,
                target_file,
                MAIN_EDITION_TEXT_FIELDS,
                class_field,
                main_edition_file,
            )

        model: FastText._FastText = FastText.load_model(model_file)

        if test_label:
            result = model.test_label(target_file.name, **kwargs)
        else:
            result = model.test(target_file.name, **kwargs)
        os.remove(target_file.name)
        return result
