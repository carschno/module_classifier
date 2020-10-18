import argparse
import csv
import logging
from collections import defaultdict
from functools import cached_property, reduce
from typing import Dict, IO, List, Sequence

ID_FIELD = "item_title"

LOGGER = logging.getLogger("merge_csv")
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.INFO)


class CSVFile:
    def __init__(
        self,
        rows: List[Dict[str, str]],
        fieldnames: Sequence[str],
        id_field: str,
    ):
        self.rows: List[Dict[str, str]] = rows
        self.fieldnames: Sequence[str] = fieldnames
        self.id_field = id_field

    @cached_property
    def ids_rows(self) -> Dict[str, Dict[str, str]]:
        return defaultdict(
            dict, {row[self.id_field]: row for row in self.rows}
        )

    def merge(self, other: 'CSVFile') -> 'CSVFile':
        assert self.id_field == other.id_field

        all_fields = list(self.fieldnames) + [
            fieldname
            for fieldname in other.fieldnames
            if fieldname not in self.fieldnames
        ]
        rows = [
            {**self.ids_rows[row_id], **other.ids_rows[row_id]}
            for row_id in set(self.ids_rows.keys()).union(
                other.ids_rows.keys()
            )
        ]
        assert len(rows) <= len(self.rows) + len(other.rows), (
            f"#rows in individual CSVs ({len(self.rows)}, {len(other.rows)}) "
            f"before merge larger than after merge ({len(rows)})."
        )
        if len(rows) >= max(len(self.rows), len(other.rows)):
            LOGGER.warning("Found some duplicate lines.")
        return CSVFile(rows, all_fields, self.id_field)

    def write(self, f: IO):
        writer = csv.DictWriter(f, self.fieldnames)
        writer.writeheader()
        writer.writerows(self.rows)
        LOGGER.info(f"Wrote {len(self.rows)} rows.")

    @classmethod
    def from_file(cls, f: IO, id_field: str):
        reader: csv.DictReader = csv.DictReader(f)
        rows = list(reader)
        LOGGER.info(f"Read {len(rows)} rows.")
        return cls(rows, reader.fieldnames, id_field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge CSV Files.')
    parser.add_argument(
        "--id-field",
        "-f",
        type=str,
        default="item_title",
        help="The column to use for identifying corresponding rows.",
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        type=argparse.FileType("r"),
        help="The input files.",
    )
    parser.add_argument(
        "--output", "-o", type=argparse.FileType("w"), help="The output file."
    )
    args = parser.parse_args()

    merged: CSVFile = reduce(
        CSVFile.merge,
        (CSVFile.from_file(f, args.id_field) for f in args.input),
    )
    merged.write(args.output)
