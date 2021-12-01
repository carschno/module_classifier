import csv
import os
from typing import Dict, Generator, Iterable, Optional, Tuple

import numpy as np

from ..classification.settings import CLASS_FIELD, TEXT_FIELDS
from ..preprocessing import fasttext_line


class Preprocessor:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file

    def read_csv(self) -> Iterable[Dict[str, str]]:
        with open(self.csv_file) as f:
            reader = csv.DictReader(f, dialect="excel")
            for row in reader:
                if row:
                    yield row

    def write_fasttext(
        self,
        target_file: str,
        split: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        suffixes=(".train", ".dev", ".test"),
        seed: Optional[int] = None,
    ):
        if not sum(split) == 1.0:
            raise ValueError("Split must sum up to 1.0")
        np.random.seed(seed)

        target_files = [target_file + suffix for suffix in suffixes]
        handlers = [open(f, "w") for f in target_files]

        for line in self.generate_fasttext_lines():
            handler = np.random.choice(handlers, p=split)
            handler.write(line)
            handler.write(os.linesep)

        for handler in handlers:
            handler.close()

    def generate_fasttext_lines(self) -> Generator[str, None, None]:
        return (
            fasttext_line(row, TEXT_FIELDS)
            for row in self.read_csv()
            if row.get(CLASS_FIELD)
        )
