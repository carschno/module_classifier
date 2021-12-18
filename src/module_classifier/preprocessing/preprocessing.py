import os

from .settings import MIN_TOKEN_LENGTH, PUNCTUATION_CHARACTERS


def clean(s: str) -> str:
    for c in PUNCTUATION_CHARACTERS:
        s = s.replace(c, " ")

    # Replace all numbers with 0
    for n in "0123456789":
        s = s.replace(n, "0")

    # Remove linebreaks, lowercase, remove short tokens
    return " ".join(
        (
            token
            for token in s.strip().replace(os.linesep, " ").lower().split()
            if len(token) >= MIN_TOKEN_LENGTH
        )
    )
