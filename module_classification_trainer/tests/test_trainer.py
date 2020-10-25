from csv import DictWriter
from tempfile import NamedTemporaryFile, TemporaryFile
from typing import Dict, List

import pytest
from module_classifier.settings import CLASS_FIELD, TEXT_FIELDS

from src.training.trainer import Trainer


@pytest.mark.parametrize(
    "input,expected", [([{}], []), ([{"field": "text"}], [])]
)
def test_write_training_file(input: List[Dict[str, str]], expected: List[str]):
    trainer = Trainer()
    fieldnames = {column for row in input for column in row.keys()}

    with NamedTemporaryFile("r+") as csvfile, NamedTemporaryFile(
        "r+t"
    ) as target_file:
        writer = DictWriter(csvfile, fieldnames)
        writer.writerows(input)
        csvfile.flush()

        trainer._write_training_file(
            csvfile.name, target_file, TEXT_FIELDS, CLASS_FIELD
        )
        assert target_file.readlines() == expected
