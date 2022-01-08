import csv
import os
from tempfile import NamedTemporaryFile

from src.module_classifier.preprocessing.archive_files import (
    ArchiveFile,
    MainEditionFile,
    merge_data,
)

from ..conftest import TEST_ARCHIVE_FILE, TEST_MAIN_EDITION_ITEMS_FILE


class TestArchiveFile:
    def test_init(self):
        archive_file = ArchiveFile(open(TEST_ARCHIVE_FILE))

        assert archive_file.fieldnames == [
            "id",
            "content_type",
            "authors",
            "publication_name",
            "external_url",
            "language",
            "item_title",
            "module_id_for_all",
            "item_status",
            "paywall",
            "excerpt_ts",
        ]
        assert archive_file.all_ids == {
            3981,
            3982,
            3983,
            3984,
            3985,
            3986,
            3987,
            3988,
            3989,
        }
        assert len(archive_file._rows) == len(archive_file.all_ids)


class TestMainEditionsFile:
    def test_init(self):
        main_editions_file = MainEditionFile(open(TEST_MAIN_EDITION_ITEMS_FILE))

        assert main_editions_file._fieldnames == [
            "syllabus_id",
            "link_id",
            "rank",
            "excerpt_override",
        ]
        assert main_editions_file._ids == {
            111494,
            110726,
            111561,
            111435,
            3985,
            3987,
            3986,
            111474,
            110903,
        }
        assert len(main_editions_file._rows) == len(main_editions_file._ids)

    def test_contains(self):
        main_editions_file = MainEditionFile(open(TEST_MAIN_EDITION_ITEMS_FILE))

        assert all(id_ in main_editions_file for id_ in main_editions_file._ids)
        assert 0 not in main_editions_file
        assert 111493 not in main_editions_file
        assert 110904 not in main_editions_file
        assert "test id" not in main_editions_file


class TestMainEditionPreprocessing:
    def test_merge_data(self):
        expected_negative_ids = {3981, 3982, 3983, 3984, 3987, 3988, 3989}
        expected_positive_ids = {3985, 3986, 3987}

        archive_file = ArchiveFile(open(TEST_ARCHIVE_FILE))
        main_editions_file = MainEditionFile(open(TEST_MAIN_EDITION_ITEMS_FILE))

        with NamedTemporaryFile("w", delete=False) as output_file:
            merge_data(archive_file, main_editions_file, output_file)

        with open(output_file.name) as output_file:
            reader = csv.DictReader(output_file)
            rows = [row for row in reader]

        assert reader.fieldnames == archive_file.fieldnames + ["label"]
        assert len(rows) == len(main_editions_file._rows)

        for row in rows:
            id_ = int(row["id"])
            assert id_ in expected_negative_ids or id_ in expected_positive_ids

            if id_ in expected_positive_ids:
                assert row["label"] == "True", f"{id_} should be 'True'"
            elif id_ in expected_negative_ids:
                assert row["label"] == "False", f"{id_} shoud be 'False'"

        os.remove(output_file.name)
