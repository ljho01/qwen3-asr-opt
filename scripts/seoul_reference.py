"""Strict reference handling for the Seoul Corpus confirmation, never model input."""
from __future__ import annotations

import math
import re

INTERVAL = re.compile(
    r'intervals \[(\d+)\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*'
    r'text = "((?:[^\"]|\"\")*)"', re.DOTALL
)
NONLEXICAL = {"SIL", "VOCNOISE", "LAUGH", "NOISE"}


def read_utterances(blob: bytes) -> list[dict]:
    """Read the unique utterance orthographic tier and verify complete time coverage."""
    text = blob.decode("utf-16" if blob.startswith((b"\xfe\xff", b"\xff\xfe")) else "utf-8-sig")
    tiers = re.split(r"^    item \[\d+\]:", text, flags=re.MULTILINE)[1:]
    selected = [t for t in tiers if re.search(r'^\s*name = "utt\.ortho\."\s*$', t, re.MULTILINE)]
    if len(selected) != 1:
        raise ValueError("Require exactly one utt.ortho. tier")
    tier = selected[0]
    count = int(re.search(r"intervals: size = (\d+)", tier).group(1))
    header = tier.split("intervals: size", 1)[0]
    lower = float(re.search(r"xmin = ([\d.e+-]+)", header).group(1))
    upper = float(re.search(r"xmax = ([\d.e+-]+)", header).group(1))
    rows = [{"index": int(i), "start": float(a), "end": float(b), "text": s.replace('""', '"')}
            for i, a, b, s in INTERVAL.findall(tier)]
    if len(rows) != count or not rows:
        raise ValueError("Incomplete TextGrid interval parse")
    previous = lower
    for index, row in enumerate(rows, 1):
        if row["index"] != index or not all(math.isfinite(row[k]) for k in ("start", "end")):
            raise ValueError("Invalid interval index or time")
        if abs(row["start"] - previous) > 1e-7 or row["end"] < row["start"]:
            raise ValueError("Reference interval gap, overlap or reversed bounds")
        previous = row["end"]
    if abs(previous - upper) > 1e-7:
        raise ValueError("Reference tier is incomplete")
    return rows


def lexical_text(text: str) -> str:
    """Drop only known nonlexical tags; preserve words spoken during laughter."""
    def marker(match):
        value = match.group(1)
        if value in NONLEXICAL:
            return " "
        if value.startswith("LAUGH-") and value[6:].strip():
            return value[6:]
        raise ValueError(f"Untranscribed or unsupported marker: <{value}>")

    result = re.sub(r"<([^<>]*)>", marker, text)
    if "<" in result or ">" in result:
        raise ValueError("Malformed reference marker")
    return " ".join(result.split())


def excerpt_reference(rows: list[dict], start: float, end: float) -> tuple[str, list[dict]]:
    """Require whole reference intervals; never truncate words to fit a time crop."""
    selected = [r for r in rows if r["end"] > start + 1e-7 and r["start"] < end - 1e-7]
    if not selected or abs(selected[0]["start"] - start) > 1e-7 or abs(selected[-1]["end"] - end) > 1e-7:
        raise ValueError("Excerpt must align with reference interval boundaries")
    texts = [lexical_text(row["text"]) for row in selected]
    return " ".join(t for t in texts if t), selected
