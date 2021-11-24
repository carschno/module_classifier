from re import S
import pytest
from src.module_classifier.models import Module


class TestModule:
    @pytest.mark.parametrize(
        "input,label_prefix,delimiters,expected",
        [
            (
                "s1.m1",
                "",
                (".",),
                Module(S=1, module=1),
            ),
            (
                "s10.m10",
                "",
                (".",),
                Module(S=10, module=10),
            ),
            (
                "s1.m1",
                "",
                (".", "_"),
                Module(S=1, module=1),
            ),
            (
                "s1_m1",
                "",
                (".", "_"),
                Module(S=1, module=1),
            ),
            (
                "S1_M1",
                "",
                (".", "_"),
                Module(S=1, module=1),
            ),
            ("__label__s1.m1", "__label__", (".", "_"), Module(S=1, module=1)),
        ],
    )
    def test_from_string(self, input, label_prefix, delimiters, expected):
        assert (
            Module.from_string(input, label_prefix=label_prefix, delimiters=delimiters)
            == expected
        )

    @pytest.mark.parametrize(
        "input,label_prefix,delimiters",
        [
            ("", "", (".",)),
            ("s1.m1", "", ("_",)),
            ("s1.m1", "__label__", (".",)),
            ("__label__s1.m1", "", (".",)),
        ],
    )
    def test_from_string_invalid(self, input, label_prefix, delimiters):
        with pytest.raises(ValueError):
            Module.from_string(input, label_prefix=label_prefix, delimiters=delimiters)

    @pytest.mark.parametrize(
        "input,label_prefix,delimiter,expected",
        [
            (Module(S=1, module=1), "__label__", "_", "__label__S1_M1"),
            (Module(S=1, module=1), "__label__", ".", "__label__S1.M1"),
            (Module(S=10, module=10), "__label__", ".", "__label__S10.M10"),
        ],
    )
    def test_fasttext(
        self, input: Module, label_prefix: str, delimiter: str, expected: str
    ):
        assert (
            input.fasttext(label_prefix=label_prefix, delimiter=delimiter) == expected
        )
