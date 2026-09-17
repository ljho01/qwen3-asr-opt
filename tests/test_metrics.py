import pytest

from qwen_asr_opt.metrics import normalize, score


def test_korean_spacing_and_punctuation():
    result = score([{"reference": "안녕 하세요.", "text": "안녕하세요!"}])
    assert result["cer_no_space"] == 0
    assert result["wer"] > 0
    assert normalize("Ａbc,   TEST!") == "abc test"


def test_corpus_weighting_not_average_of_clip_rates():
    result = score([{"reference": "one", "text": "two"},
                    {"reference": "one two three four", "text": "one two three four"}])
    assert result["wer"] == pytest.approx(0.2)
    assert result["reference_words"] == 5


def test_silence_hallucination_counts_as_insertion():
    result = score([{"reference": "", "text": "hello"}])
    assert result["word_errors"] == 1
    assert result["reference_words"] == 0
