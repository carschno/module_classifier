#!/usr/bin/env python

"""
Example call:
/opt/anaconda3/envs/syllabus/bin/python /home/carsten/syllabus/workspace/module-classifier-api/tools/prepare_data.py -i for_retraining_url_and_module.csv general_archive.csv --id external_url | /opt/anaconda3/envs/syllabus/bin/python /home/carsten/syllabus/workspace/module-classifier-api/tools/prepare_data.py -i - yt_description.csv --id external_url --keep-unmatched --split 0.8 0.1 0.1 -o merged.csv --seed 0
"""
import argparse
import csv
import logging
import random
import sys
from functools import reduce
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO)


def split(
    rows: List[Dict[str, Any]], ratio: List[float], seed: Optional[int]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into training, validation and test sets.

    Args:
        rows (List[Dict[str, Any]]): all rows, each represented as a dict
        ratio (List[float]): the train/dev/test split ratio, e.g. [0.8, 0.1, 0.1]
        seed (Optional[int]): a random seed to use for shuffling, if given

    Raises:
        ValueError: if the 'ratio' list is not of length 3, or does not sum up to 1.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]: The input rows split into training, validation and test sets.
    """
    if len(ratio) != 3:
        raise ValueError("ratio must be a list of 3 floats")
    if sum(ratio) != 1:
        raise ValueError("Ratio must sum to 1")

    if seed is not None:
        logging.info(f"Using random seed '{seed}' for shuffling train/dev/test split.")
        random.seed(seed)

    random.shuffle(rows)

    train_size = int(len(rows) * ratio[0])
    dev_size = int(len(rows) * ratio[1])

    train = rows[:train_size]
    dev = rows[train_size : train_size + dev_size]
    test = rows[train_size + dev_size :]

    return train, dev, test


def _merge_rows(
    rows1: List[Dict[str, Any]], rows2: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge rows with the same key.

    Note: reads args from outer scope.
    """

    key = args.id

    index2 = {row[key]: row for row in rows2}

    merged = []
    for row in rows1:
        match: Optional[Dict[str, Any]] = index2.get(row[key])

        if match is None:
            if args.keep_unmatched:
                merged.append(row)
            else:
                logging.warning(
                    f"No matching row found for '{key}' '{row[key]}'. Skipping"
                )
        else:
            merged.append(dict(row, **match))
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge files with content and labels, split into sets."
    )

    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        type=argparse.FileType("r"),
        help="Input files",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=sys.stdout,
        type=argparse.FileType("w"),
        help="Output file. Defaults to stdout.",
    )

    parser.add_argument(
        "--id", required=True, type=str, help="ID column across all input files."
    )

    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        required=False,
        help="If given, split into train/test/dev. Must be 3 numbers summing up to 1.0. Default: do not split, write everything to the output file.",
    )

    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        help="If given, train/dev/test split is randomized with this seed.",
    )

    parser.add_argument(
        "--keep-unmatched", action="store_true", help="Keep unmatched rows."
    )

    args = parser.parse_args()

    if args.split is not None and args.output is sys.stdout:
        raise ValueError("Cannot split to stdout")

    if len(args.input) < 2:
        raise ValueError("Must have at least 2 input files")

    input_data: List[List[Dict[str, Any]]] = []

    for f in args.input:
        logging.info(f"Reading from'{f.name}'...")
        reader = csv.DictReader(f)
        if reader.fieldnames and args.id in reader.fieldnames:
            input_data.append([line for line in reader])
        else:
            raise ValueError(f"'{f.name}' does not contain column '{args.id}'.")

    merged: List[Dict[str, Any]] = reduce(_merge_rows, input_data)
    fieldnames: Set[str] = {key for line in merged for key in line.keys()}

    if args.split is None:
        outputs = [(merged, args.output)]
    else:
        train, dev, test = split(merged, args.split, args.seed)

        outputs = [
            (train, args.output),
            (dev, open(args.output.name + ".dev.csv", "w")),
            (test, open(args.output.name + ".test.csv", "w")),
        ]

    for lines, f in outputs:
        logging.info(f"Writing {len(lines)} rows to '{f.name}'...")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
