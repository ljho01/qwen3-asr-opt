import mlx.core as mx
import pytest
from mlx_qwen3_asr.encoder import AudioEncoderLayer, _create_windowed_mask

from qwen_asr_opt.encoder import batched_window_layers


@pytest.mark.parametrize("boundaries", [[0, 7], [0, 8, 16, 19], [0, 3, 10, 11]])
def test_padded_windows_match_dense_and_isolate_windows(boundaries):
    mx.random.seed(13)
    layers = [AudioEncoderLayer(64, 2, 128) for _ in range(2)]
    x = mx.random.normal((1, boundaries[-1], 64))
    mask = _create_windowed_mask(x.shape[1], boundaries)
    expected = x
    for layer in layers:
        expected = layer(expected, mask=mask)
    result = batched_window_layers(x, layers, boundaries)
    assert mx.allclose(result, expected, atol=3e-6, rtol=3e-6).item()
    if len(boundaries) > 2:
        # A neighboring window cannot change the first window's features.
        perturbed = mx.concatenate([x[:, :boundaries[1]], x[:, boundaries[1]:] + 50], axis=1)
        changed = batched_window_layers(perturbed, layers, boundaries)
        assert mx.array_equal(result[:, :boundaries[1]], changed[:, :boundaries[1]]).item()


@pytest.mark.parametrize("boundaries", [[], [0], [0, 0, 7], [0, 8], [1, 7]])
def test_invalid_windows_are_rejected(boundaries):
    with pytest.raises(ValueError):
        batched_window_layers(mx.zeros((1, 7, 64)), [], boundaries)
