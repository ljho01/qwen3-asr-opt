"""Strict local-only references for MagicHub's Korean conversational corpus."""
from __future__ import annotations

import math
import re

NONLEXICAL = {"LAUGHTER", "SONANT", "MUSIC", "SYSTEM", "ENS"}
LINE = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+\S+\s+(.*)")


def lexical_text(text: str) -> str:
    """Remove documented noise markers only; never repair words or numbers."""
    def replace(match):
        if match[1] not in NONLEXICAL:
            raise ValueError(f"Unknown or untranscribed marker: {match[0]}")
        return " "

    text = re.sub(r"\[([^\[\]]*)\]", replace, text)
    if "[" in text or "]" in text:
        raise ValueError("Malformed reference marker")
    # The selected pair has no overlap '+' markers. Refuse ambiguous attached '+'.
    words = text.split()
    if any("+" in word and word != "+" for word in words):
        raise ValueError("Ambiguous attached overlap marker")
    return " ".join(word for word in words if word != "+")


def read_reference(blob: bytes, speaker: str, duration_s: float) -> list[dict]:
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Invalid audio duration")
    rows = []
    previous = -1.0
    for index, line in enumerate(blob.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        match = LINE.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"Unparsed reference line {index}")
        start, end = float(match[1]), float(match[2])
        if not all(math.isfinite(v) for v in (start, end)):
            raise ValueError("Nonfinite reference timestamp")
        if not 0 <= start < end <= duration_s or start < previous:
            raise ValueError("Reference time order or bounds are invalid")
        text = lexical_text(match[4])
        if match[3] != speaker and not (match[3] == "0" and not text):
            raise ValueError("Unexpected speaker or unassigned lexical speech")
        rows.append({"line": index, "start": start, "end": end,
                     "speaker": match[3], "text": text})
        previous = start
    if not rows or not any(row["text"] for row in rows):
        raise ValueError("Empty lexical reference")
    return rows


def serialize(rows: list[dict]) -> str:
    """Fixed onset ordering; overlap is retained, not an overlap-invariant metric."""
    return " ".join(row["text"] for row in sorted(
        rows, key=lambda row: (row["start"], row["end"], row["speaker"], row["line"])
    ) if row["text"])
