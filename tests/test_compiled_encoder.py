import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from mlx_qwen3_asr.config import AudioEncoderConfig
from mlx_qwen3_asr.encoder import AudioEncoder

from qwen_asr_opt.compiled_encoder import NativeCompiledEncoder


def encoder():
    mx.random.seed(3107)
    model = AudioEncoder(AudioEncoderConfig(encoder_layers=2, encoder_attention_heads=4,
        encoder_ffn_dim=128, d_model=64, output_dim=128, downsample_hidden_size=8))
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=64, bits=8)
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("frames", [1, 99, 100, 101, 799, 800, 801])
def test_tail_and_window_boundaries_match_upstream(frames):
    original = encoder()
    wrapper = NativeCompiledEncoder(original)
    mel = mx.random.normal((1, 128, frames)).astype(mx.float16)
    lengths = mx.array([frames])
    expected, expected_lengths = original(mel, lengths)
    actual, actual_lengths = wrapper(mel, lengths)
    mx.eval(expected, actual)
    a, b = np.asarray(actual).astype(np.float32), np.asarray(expected).astype(np.float32)
    assert a.shape == b.shape and np.isfinite(a).all()
    assert np.linalg.norm(a-b) / np.linalg.norm(b) < 0.004
    assert mx.array_equal(actual_lengths, expected_lengths).item()
    assert mx.array_equal(actual_lengths, wrapper.get_output_lengths(lengths)).item()


def test_padding_is_excluded_and_only_output_padding_added():
    original = encoder()
    wrapper = NativeCompiledEncoder(original)
    mel = mx.random.normal((2, 128, 801)).astype(mx.float16)
    lengths = mx.array([101, 801])
    expected, expected_lengths = original(mel, lengths)
    actual, actual_lengths = wrapper(mel, lengths)
    changed = mx.concatenate([mel[:1, :, :101], mx.full((1, 128, 700), 900, mx.float16)], axis=2)
    perturbed, _ = wrapper(mx.concatenate([changed, mel[1:]], axis=0), lengths)
    assert mx.array_equal(actual, perturbed).item()
    assert mx.array_equal(actual_lengths, expected_lengths).item()
    assert mx.all(actual[0, expected_lengths[0].item():] == 0).item()
    assert mx.allclose(actual, expected, atol=0.005, rtol=0.005).item()


def test_lru_reuses_recent_shape_and_bounds_callables_without_copying_weights():
    original = encoder()
    wrapper = NativeCompiledEncoder(original, max_graphs=3)
    for frames in (101, 111, 101, 121, 131, 111):
        mx.eval(wrapper(mx.zeros((1, 128, frames), mx.float16), mx.array([frames])))
        assert len(wrapper.graphs) <= 3
    assert wrapper.statistics() == {"calls": 6, "graphs_created": 5, "cache_hits": 1,
                                   "evictions": 2, "retained_graphs": 3, "max_graphs": 3}
    assert wrapper.original is original
    assert [id(v) for _, v in tree_flatten(original.parameters())] == [
        id(v) for _, v in tree_flatten(wrapper.parameters())]


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_cache_bound(count):
    with pytest.raises(ValueError):
        NativeCompiledEncoder(encoder(), max_graphs=count)


@pytest.mark.parametrize("shape,lens", [((0, 128, 100), []), ((1, 64, 100), [100]),
    ((1, 128, 100), [0]), ((1, 128, 100), [101]), ((1, 128, 100), [1.5]),
    ((2, 128, 100), [100]), ((2, 128, 100), [100, -1])])
def test_invalid_inputs_fail_before_graph_creation(shape, lens):
    wrapper = NativeCompiledEncoder(encoder())
    with pytest.raises(ValueError):
        wrapper(mx.zeros(shape, mx.float16), mx.array(lens))
    assert wrapper.calls == wrapper.graphs_created == 0
