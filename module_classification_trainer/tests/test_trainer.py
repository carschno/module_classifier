import logging
import os
from contextlib import contextmanager
from csv import DictWriter
from tempfile import NamedTemporaryFile
from typing import Dict, List

import pytest
from fasttext import FastText
from module_classifier.settings import CLASS_FIELD, TEXT_FIELDS

from src.module_trainer.settings import TRAINING_PARAMS
from src.module_trainer.training.trainer import Trainer


@contextmanager
def does_not_raise():
    yield


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
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                }
            ],
            [
                "__label__TEST_MODULE test title test authors test publication test abstract test excerpt test description"
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt two",
                    "yt_description": "test yt description two",
                },
            ],
            [
                "__label__TEST_MODULE test title test authors test publication test abstract test excerpt test description",
                "__label__TEST_MODULE test title two test authors two test publication two test abstract two test excerpt two test description two",
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                }
            ]
            * 20,
            [
                "__label__TEST_MODULE test title test authors test publication test abstract test excerpt test description"
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
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt two",
                    "yt_description": "test yt description two",
                },
            ],
            False,
            ["__label__TEST_MODULE"],
            [
                "test",
                "two",
                "title",
                "authors",
                "publication",
                "abstract",
                "excerpt",
                "description",
                "</s>",
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "module_id_for_all": "test_module",
                    "excerpts_ts": "test excerpt two",
                    "yt_description": "test yt description two",
                },
            ],
            True,
            ["__label__TEST_MODULE"],
            [
                "test",
                "two",
                "title",
                "authors",
                "publication",
                "abstract",
                "excerpt",
                "description",
                "</s>",
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "module_id_for_all": "test_module_1",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "module_id_for_all": "test_module_2",
                    "excerpts_ts": "test excerpt two",
                    "yt_description": "test yt description two",
                },
            ],
            False,
            ["__label__TEST_MODULE_1", "__label__TEST_MODULE_2"],
            [
                "test",
                "two",
                "title",
                "authors",
                "publication",
                "abstract",
                "excerpt",
                "description",
                "</s>",
            ],
        ),
    ],
)
def test_train_model(caplog, input, quantize, expected_labels, expected_words):
    trainer = Trainer(cpus=1, training_parameters={**TRAINING_PARAMS, "epoch": 1})

    with NamedTemporaryFile("w") as csvfile, NamedTemporaryFile(
        delete=False
    ) as target_file, caplog.at_level(logging.INFO, logger=trainer.__class__.__name__):
        writer = DictWriter(csvfile, {column for row in input for column in row.keys()})
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
    assert sorted(model.labels) == sorted(expected_labels)
    assert sorted(model.words) == sorted(expected_words)
    os.remove(out)


@pytest.mark.parametrize(
    "cpus,expected_exception",
    [(1, does_not_raise()), (1000000, pytest.raises(ValueError))],
)
def test_cpus(cpus, expected_exception):
    with expected_exception:
        Trainer(cpus=cpus)
