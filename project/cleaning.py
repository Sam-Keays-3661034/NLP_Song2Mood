import re
from typing import List

# Regexes shared across models
WORD_TOKENIZE_PATTERN = re.compile(r"(?u)\b\w\w+\b")
TIMESTAMP_PATTERN = re.compile(r"\[\d{2}:\d{2}(?:\.\d{2})?\]")


def word_tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used by classical baselines."""
    return [token.lower() for token in WORD_TOKENIZE_PATTERN.findall(text)]


def clean_lyrics(text: str) -> str:
    """Normalize lyric text for downstream models."""
    text = TIMESTAMP_PATTERN.sub("", text)
    text = text.lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"_{2,}", " ", text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


__all__ = ["clean_lyrics", "word_tokenize"]
