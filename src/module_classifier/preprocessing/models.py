import re
from typing import Iterable, Match, Optional, Pattern

from pydantic import BaseModel, PositiveInt

from .settings import DEFAULT_MODULE_DELIMITER


class Module(BaseModel):
    """Representation of the 'module' as defined in the taxonomy."""

    section: PositiveInt
    module: PositiveInt

    def __str__(self, delimiter=DEFAULT_MODULE_DELIMITER):
        return f"S{self.section}{delimiter}M{self.module}"

    def fasttext(
        self, label_prefix: str, delimiter: str = DEFAULT_MODULE_DELIMITER
    ) -> str:
        """Generate a string representation suitable for a FastText line."""
        return f"{label_prefix}{self.__str__(delimiter)}"

    @classmethod
    def from_string(cls, s: str, *, label_prefix: str = "", delimiters: Iterable[str]):
        """Parse a string representation of a module, optionally with a prefix (e.g. '__label__')."""

        delimiters_pattern: str = "|".join(delimiters)
        pattern: Pattern = re.compile(
            rf"^{label_prefix}(?:S|s)(?P<section>\d\d?)({delimiters_pattern})(?:M|m)(?P<module>\d\d?)$"
        )

        match: Optional[Match] = pattern.fullmatch(s)

        if match is None:
            raise ValueError(f"Invalid module string: {s}")
        else:
            return cls(
                section=int(match.group("section")),
                module=int(match.group("module")),
            )
