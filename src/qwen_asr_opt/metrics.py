"""Explicit, language-neutral scoring; no hidden number expansion or translation."""
from __future__ import annotations

import unicodedata

import jiwer


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    # Remove punctuation but preserve letters, numbers and word spacing.
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def score(records: list[dict]) -> dict:
    if not records:
        return {"samples": 0, "wer": None, "cer": None, "cer_no_space": None}
    refs = [normalize(r["reference"]) for r in records]
    hyps = [normalize(r["text"]) for r in records]
    words = jiwer.process_words(refs, hyps)
    chars = jiwer.process_characters(refs, hyps)
    compact = jiwer.process_characters(
        [s.replace(" ", "") for s in refs], [s.replace(" ", "") for s in hyps]
    )
    return {
        "samples": len(records), "wer": words.wer, "cer": chars.cer,
        "cer_no_space": compact.cer,
        "word_errors": words.substitutions + words.deletions + words.insertions,
        "reference_words": words.hits + words.substitutions + words.deletions,
        "char_errors_no_space": compact.substitutions + compact.deletions + compact.insertions,
        "reference_chars_no_space": compact.hits + compact.substitutions + compact.deletions,
    }
