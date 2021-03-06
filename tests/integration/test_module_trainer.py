import logging
import os
from csv import DictWriter
from distutils.util import strtobool
from tempfile import NamedTemporaryFile
from typing import Dict, List

import pytest
from fasttext import FastText
from src.module_classifier.preprocessing.settings import CLASS_FIELD, TEXT_FIELDS
from src.module_classifier.training.module_trainer import ModuleTrainer
from src.module_classifier.training.settings import TRAINING_PARAMS

from ..conftest import does_not_raise

TEST_QUANTIZATION: bool = strtobool(os.environ.get("TEST_QUANTIZATION", "False"))


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
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                }
            ],
            [
                "__label__S1_M1 test title test authors test publication test abstract test excerpt test description"
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt two",
                    "yt_description": "test yt description two",
                },
            ],
            [
                "__label__S1_M1 test title test authors test publication test abstract test excerpt test description",
                "__label__S1_M1 test title two test authors two test publication two test abstract two test excerpt two test description two",
            ],
        ),
        (
            [
                {
                    "item_title": "test title",
                    "authors": "test authors",
                    "publication_name": "test publication",
                    "abstract_description": "test abstract",
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                }
            ]
            * 20,
            [
                "__label__S1_M1 test title test authors test publication test abstract test excerpt test description"
            ]
            * 20,
        ),
    ],
)
def test_write_training_file(input: List[Dict[str, str]], expected: List[str]):
    trainer = ModuleTrainer()
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
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt two",
                    "yt_description": "test yt description two",
                },
            ],
            False,
            ["__label__S1_M1"],
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
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt",
                    "yt_description": "test yt description",
                },
                {
                    "item_title": "test title two",
                    "authors": "test authors two",
                    "publication_name": "test publication two",
                    "abstract_description": "test abstract two",
                    "module_id_for_all": "s1.m1",
                    "excerpts_ts": "test excerpt two",
                    "yt_description": "test yt description two",
                },
            ],
            True,
            ["__label__S1_M1"],
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
    if quantize and not TEST_QUANTIZATION:
        pytest.skip("Skipping quantization test case.")

    trainer = ModuleTrainer(cpus=1, training_parameters={**TRAINING_PARAMS, "epoch": 1})

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
        if quantize:
            assert "Compressing model..." in caplog.messages
    model: FastText = FastText.load_model(out)
    assert model.is_quantized() == quantize
    assert sorted(model.labels) == sorted(expected_labels)
    assert sorted(model.words) == sorted(expected_words)
    os.remove(out)
