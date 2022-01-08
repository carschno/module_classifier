import csv
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, TextIO


class ArchiveFile:
    ID_FIELD: str = "id"

    def __init__(self, archive_file: TextIO):
        self._read_file(archive_file)

    def _read_file(self, archive_file: TextIO):
        reader = csv.DictReader(archive_file)
        self._fieldnames: Optional[Sequence[str]] = reader.fieldnames
        self._rows: List[Dict[str, Any]] = [row for row in reader]
        self._ids: Set[int] = {int(row[self.ID_FIELD]) for row in self._rows}

    @property
    def fieldnames(self) -> Sequence[str]:
        if self._fieldnames is None:
            raise RuntimeError("Archive file has not been read yet.")
        return self._fieldnames

    @property
    def all_ids(self) -> Set[int]:
        return self._ids

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._rows)


class MainEditionFile:
    def __init__(self, main_edition_file: TextIO) -> None:
        self._read_file(main_edition_file)

    def _read_file(self, file: TextIO, id_field: str = "link_id"):
        reader = csv.DictReader(file)
        self._fieldnames: Optional[Sequence[str]] = reader.fieldnames
        self._rows: List[Dict[str, Any]] = [row for row in reader]
        self._ids: Set[int] = {int(row[id_field]) for row in self._rows}

    def __contains__(self, key):
        return key in self._ids


def merge_data(
    archive_file: ArchiveFile,
    main_edition_file: MainEditionFile,
    output_file: TextIO,
    label_field: str = "label",
):
    """Merge two data files.

    Args:
        archive_file: the archive file
        main_edition_file: the main edition file
        output_file: the output file
    """
    writer = csv.DictWriter(
        output_file, fieldnames=list(archive_file.fieldnames) + [label_field]
    )
    writer.writeheader()

    for row in archive_file:
        label: bool = int(row["id"]) in main_edition_file
        writer.writerow({**row, "label": label})
