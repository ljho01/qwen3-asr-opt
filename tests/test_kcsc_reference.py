import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "kcsc_reference", Path(__file__).resolve().parents[1] / "scripts/kcsc_reference.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_words_fillers_partial_words_and_numbers_are_preserved():
    rows = module.read_reference(
        b"[0,1] A x uh 13 par- [LAUGHTER]\n[1,2] 0 x [ENS]\n[2,3] A x + yes",
        "A", 3,
    )
    assert module.serialize(rows) == "uh 13 par- yes"


@pytest.mark.parametrize("line", [
    "[0,1] A x [*]", "[0,1] A x [UNKNOWN]", "[0,1] A x [ENS", "[0,1] A x yes+",
    "[0,1] B x yes", "[0,1] 0 x yes", "[0,4] A x yes", "[1,0] A x yes",
    "[0,1] A x yes\n[.5,.6] A x no\n[.1,.2] A x uh", "bad line",
])
def test_incomplete_or_ambiguous_reference_rejected(line):
    with pytest.raises(ValueError):
        module.read_reference(line.encode(), "A", 3)


def test_overlap_retained_with_fixed_onset_order():
    a = module.read_reference(b"[0,2] A x first\n[2,3] A x third", "A", 3)
    b = module.read_reference(b"[1,2.5] B x second", "B", 3)
    assert module.serialize(b + a) == "first second third"
