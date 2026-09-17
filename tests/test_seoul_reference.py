import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("seoul_reference", Path(__file__).resolve().parents[1] / "scripts/seoul_reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def grid():
    return '''File type = "ooTextFile"
Object class = "TextGrid"
    item [1]:
        class = "IntervalTier"
        name = "utt.ortho."
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "그는 ""네""라고"
        intervals [2]:
            xmin = 1
            xmax = 2
            text = "<LAUGH-말했어요> <SIL>"
'''


def test_utf16_quoted_speech_and_laughter_remain_in_reference():
    rows = module.read_utterances(grid().encode("utf-16"))
    text, selected = module.excerpt_reference(rows, 0, 2)
    assert text == '그는 "네"라고 말했어요'
    assert len(selected) == 2


@pytest.mark.parametrize("tag", ["<IVER>", "<UNKNOWN>", "<PRIVATE.INFO>", "<OTHER>", "<LAUGH->", "<LAUGH-<SIL>>"])
def test_missing_speech_cannot_silently_enter_reference(tag):
    with pytest.raises(ValueError):
        module.lexical_text("말 " + tag)


def test_incomplete_or_gapped_tier_is_rejected():
    for text in (grid().replace("size = 2", "size = 3"), grid().replace("xmin = 1", "xmin = 1.1")):
        with pytest.raises(ValueError):
            module.read_utterances(text.encode())


def test_crop_cannot_remove_part_of_reference_utterance():
    rows = module.read_utterances(grid().encode())
    with pytest.raises(ValueError, match="boundaries"):
        module.excerpt_reference(rows, 0.5, 2)
