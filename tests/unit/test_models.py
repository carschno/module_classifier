import pytest
from src.module_classifier.preprocessing import Module


class TestModule:
    @pytest.mark.parametrize(
        "input,label_prefix,delimiters,expected",
        [
            (
                "s1.m1",
                "",
                (".",),
                Module(section=1, module=1),
            ),
            (
                "s10.m10",
                "",
                (".",),
                Module(section=10, module=10),
            ),
            (
                "s1.m1",
                "",
                (".", "_"),
                Module(section=1, module=1),
            ),
            (
                "s1_m1",
                "",
                (".", "_"),
                Module(section=1, module=1),
            ),
            (
                "S1_M1",
                "",
                (".", "_"),
                Module(section=1, module=1),
            ),
            ("__label__s1.m1", "__label__", (".", "_"), Module(section=1, module=1)),
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
        "input,delimiter,expected",
        [
            (Module(section=1, module=1), "_", "S1_M1"),
            (Module(section=1, module=1), ".", "S1.M1"),
            (Module(section=10, module=10), ".", "S10.M10"),
        ],
    )
    def test_to_string(self, input: Module, delimiter: str, expected: str):
        assert input.to_string(delimiter=delimiter) == expected
