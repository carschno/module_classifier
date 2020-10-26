import logging
import os
from csv import DictWriter
from tempfile import NamedTemporaryFile
from typing import Dict, List

import pytest
from fasttext import FastText
from module_classifier.settings import CLASS_FIELD, TEXT_FIELDS

from src.settings import TRAINING_PARAMS
from src.training.trainer import Trainer


@pytest.mark.parametrize(
    "input,expected",
    [
        ([{}], []),
        ([{"field": "text"}], []),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "full_text": "test full text",
                    "module_id_for_all": "test_module",
                }
            ],
            [
                "__label__TEST_MODULE test title test authors test publication test abstract test full text"
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "full_text": "test full text",
                    "module_id_for_all": "test_module",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "full_text": "test full text two",
                    "module_id_for_all": "test_module",
                },
            ],
            [
                "__label__TEST_MODULE test title test authors test publication test abstract test full text",
                "__label__TEST_MODULE test title two test authors two test publication two test abstract two test full text two",
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "full_text": "test full text",
                    "module_id_for_all": "test_module",
                }
            ]
            * 20,
            [
                "__label__TEST_MODULE test title test authors test publication test abstract test full text"
            ]
            * 20,
        ),
    ],
)
def test_write_training_file(input: List[Dict[str, str]], expected: List[str]):
    trainer = Trainer()
    fieldnames = {column for row in input for column in row.keys()}

    with NamedTemporaryFile("w") as csvfile, NamedTemporaryFile(
        "wt", delete=False
    ) as target_file:
        writer = DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerows(input)
        csvfile.flush()

        trainer._write_training_file(
            csvfile.name, target_file, TEXT_FIELDS, CLASS_FIELD
        )
        out = target_file.name
    with open(out) as f:
        assert f.readlines() == [line + os.linesep for line in expected]
    os.remove(out)


@pytest.mark.parametrize(
    "input,quantize,expected_labels,expected_words",
    [
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "full_text": "test full text",
                    "module_id_for_all": "test_module",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "full_text": "test full text two",
                    "module_id_for_all": "test_module",
                },
            ],
            False,
            ["__label__TEST_MODULE"],
            [
                'test',
                'two',
                'title',
                'authors',
                'publication',
                'abstract',
                'full',
                'text',
                '</s>',
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "full_text": "test full text",
                    "module_id_for_all": "test_module",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "full_text": "test full text two",
                    "module_id_for_all": "test_module",
                },
            ],
            True,
            ["__label__TEST_MODULE"],
            [
                'test',
                'two',
                'title',
                'authors',
                'publication',
                'abstract',
                'full',
                'text',
                '</s>',
            ],
        ),
    ],
)
def test_train_model(caplog, input, quantize, expected_labels, expected_words):
    trainer = Trainer({**TRAINING_PARAMS, "epoch": 1})

    with NamedTemporaryFile("w") as csvfile, NamedTemporaryFile(
        delete=False
    ) as target_file, caplog.at_level(
        logging.INFO, logger=trainer.__class__.__name__
    ):
        writer = DictWriter(
            csvfile, {column for row in input for column in row.keys()}
        )
        writer.writeheader()
        writer.writerows(input)
        csvfile.flush()

        out = target_file.name
        trainer.train_model(csvfile.name, out, quantize=quantize)
        assert f"Reading input file '{csvfile.name}'..." in caplog.messages
        assert "Training model..." in caplog.messages
        if quantize:
            assert "Compressing model..." in caplog.messages
    model: FastText = FastText.load_model(out)
    assert model.is_quantized() == quantize
    assert model.labels == expected_labels
    assert model.words == expected_words
    os.remove(out)
