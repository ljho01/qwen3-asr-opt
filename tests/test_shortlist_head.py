import mlx.core as mx
import numpy as np
import pytest

from qwen_asr_opt.shortlist_head import ShortlistHead


@pytest.mark.parametrize("method", ["take", "gather", "shared"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("shape", [(4, 128), (2, 2, 128)])
def test_rescores_selected_q8_weights_and_preserves_full_vocabulary_indices(method, dtype, shape):
    mx.random.seed(283)
    dense = mx.random.normal((256, 128)).astype(dtype)
    q8 = mx.quantize(dense, group_size=64, bits=8)
    q4 = mx.quantize(mx.dequantize(*q8, group_size=64, bits=8), group_size=64, bits=4)
    head = ShortlistHead(q8, q4, candidates=8, method=method)
    x = mx.random.normal(shape).astype(dtype)
    selected = head.select(x)
    indices, actual_scores = head.rescore(x, selected)
    original = mx.quantized_matmul(x.reshape(-1, 128), *q8, bits=8, group_size=64)
    expected_scores = mx.take_along_axis(original, indices, axis=-1)
    np.testing.assert_allclose(np.array(actual_scores), np.array(expected_scores), atol=.04, rtol=.002)
    for call in (head, mx.compile(head)):
        scores = call(x).reshape(-1, 256)
        allowed = np.zeros((4, 256), dtype=bool)
        np.put_along_axis(allowed, np.array(indices), True, axis=-1)
        assert np.array_equal(np.isfinite(np.array(scores)), allowed)
        np.testing.assert_allclose(np.array(mx.take_along_axis(scores, indices, axis=-1)),
                                   np.array(actual_scores), atol=1e-5, rtol=1e-5)
        expected_tokens = np.min(np.where(np.array(actual_scores) == np.array(actual_scores).max(-1, keepdims=True),
                                          np.array(indices), 2**31-1), axis=-1)
        np.testing.assert_array_equal(np.array(mx.argmax(scores, axis=-1)), expected_tokens)


@pytest.mark.parametrize("method", ["take", "gather", "shared"])
def test_full_vocabulary_equal_scores_choose_lowest_token_id(method):
    dense = mx.ones((128, 128), dtype=mx.float16)
    q8, q4 = (mx.quantize(dense, group_size=64, bits=bits) for bits in (8, 4))
    head = ShortlistHead(q8, q4, candidates=128, method=method)
    result = mx.compile(head)(mx.ones((2, 128), dtype=mx.float16))
    np.testing.assert_array_equal(np.array(mx.argmax(result, axis=-1)), [0, 0])


def test_shortlist_can_omit_original_winner_and_is_not_exact():
    original = mx.zeros((128, 128), dtype=mx.float16)
    original[7] = 2
    original[5] = 1
    coarse = mx.zeros_like(original)
    coarse[5] = 1
    q8, q4 = mx.quantize(original, group_size=64, bits=8), mx.quantize(coarse, group_size=64, bits=4)
    head = ShortlistHead(q8, q4, candidates=1)
    x = mx.ones((1, 128), dtype=mx.float16)
    assert int(mx.argmax(mx.quantized_matmul(x, *q8, bits=8, group_size=64)).item()) == 7
    assert int(mx.argmax(head(x)).item()) == 5


@pytest.mark.parametrize("candidates", [0, 129, True])
def test_invalid_candidate_count(candidates):
    dense = mx.zeros((128, 128), dtype=mx.float16)
    q8, q4 = (mx.quantize(dense, group_size=64, bits=bits) for bits in (8, 4))
    with pytest.raises(ValueError):
        ShortlistHead(q8, q4, candidates=candidates)
