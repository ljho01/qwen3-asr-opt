from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info

from qwen_asr_opt.decode import generate_pipelined


class FakeModel:
    def __init__(self, sequence):
        self.sequence = sequence
        self.index = 0

    def create_cache(self, max_seq_len):
        return SimpleNamespace(keys=[], values=[])

    def logits(self):
        token = self.sequence[min(self.index, len(self.sequence) - 1)]
        self.index += 1
        return mx.where(mx.arange(32) == token, 10.0, -10.0)[None, None, :]

    def prefill(self, *args, **kwargs):
        return self.logits()

    def step(self, *args, **kwargs):
        return self.logits()


@pytest.mark.parametrize("sequence,cap", [([31], 20), ([2, 3, 31], 20),
                                         ([2, 3, 4, 31], 2), ([2, 31], 2),
                                         ([2] * 30, 30), ([2] * 20, 20), ([31], 0)])
def test_matches_reference_termination(sequence, cap):
    kwargs = {"input_ids": mx.array([[1]]), "audio_features": mx.zeros((1, 1, 1)),
              "position_ids": mx.zeros((1, 3, 1), dtype=mx.int32),
              "config": GenerationConfig(max_new_tokens=cap, eos_token_ids=[31])}
    expected = generate_with_info(FakeModel(sequence), **kwargs)
    actual = generate_pipelined(FakeModel(sequence), **kwargs)
    assert actual == expected
