import mlx.core as mx
import pytest
from mlx import nn
from mlx_qwen3_asr.config import AudioEncoderConfig, Qwen3ASRConfig, TextDecoderConfig
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from mlx_qwen3_asr.model import Qwen3ASRModel

from qwen_asr_opt.batch import generate_batch, prefill_batched, prefill_serial
from qwen_asr_opt.decode import generate_pipelined
from qwen_asr_opt.prefill import configure_dense_prefill


@pytest.mark.parametrize("prefix_length", [3, 7])
def test_compiled_real_attention_cache_matches_uncompiled(prefix_length):
    mx.random.seed(7)
    config = Qwen3ASRConfig(
        audio_token_id=5,
        audio_config=AudioEncoderConfig(d_model=128, output_dim=128, encoder_layers=1,
                                       encoder_attention_heads=2, encoder_ffn_dim=256,
                                       downsample_hidden_size=8),
        text_config=TextDecoderConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=128),
    )
    model = Qwen3ASRModel(config)
    model.eval()
    mx.eval(model.parameters())
    ids = mx.array([[1, 5] + [2] * (prefix_length - 2)])
    pos = mx.broadcast_to(mx.arange(prefix_length)[None, None, :], (1, 3, prefix_length))
    kwargs = {"input_ids": ids, "audio_features": mx.random.normal((1, 1, 128)),
              "position_ids": pos,
              "config": GenerationConfig(max_new_tokens=10, eos_token_ids=[63])}
    expected = generate_with_info(model, **kwargs)
    for _ in range(2):
        assert generate_pipelined(model, **kwargs, compiled=True) == expected

    # Different prompt lengths and independent token caps exercise padding masks,
    # per-row RoPE positions and completion isolation.
    other_ids = mx.array([[1, 5, 5, 4, 6, 7, 8, 9, 10]])
    other_positions = mx.broadcast_to(mx.arange(9)[None, None, :], (1, 3, 9))
    other_features = mx.random.normal((1, 2, 128))
    other_config = GenerationConfig(max_new_tokens=6, eos_token_ids=[63])
    other = generate_with_info(model, other_ids, other_features, other_positions, other_config)
    batched = generate_batch(model,
                             [(ids, kwargs["audio_features"], pos),
                              (other_ids, other_features, other_positions)],
                             [kwargs["config"], other_config])
    assert batched == [expected, other]
    prompts = [(ids, kwargs["audio_features"], pos), (other_ids, other_features, other_positions)]
    assert generate_batch(model, prompts, [kwargs["config"], other_config],
                          prefill_mode="batched") == [expected, other]
    # Compare every VALID KV entry, which also catches wrong RoPE/padding before
    # a token difference appears. Padding entries need not match (they are masked).
    _, serial_keys, serial_values = prefill_serial(model, prompts, 32)
    _, batch_keys, batch_values = prefill_batched(model, prompts, 32)
    for left, right in zip(serial_keys + serial_values, batch_keys + batch_values, strict=True):
        for row, (prompt_ids, _, _) in enumerate(prompts):
            length = prompt_ids.shape[1]
            assert mx.allclose(left[row, :, :length], right[row, :, :length],
                               atol=5e-6, rtol=5e-6).item()


@pytest.mark.parametrize("mode", ["transient", "cached"])
def test_dense_prefill_keeps_quantized_decode_and_independent_streams(mode):
    mx.random.seed(11)
    config = Qwen3ASRConfig(
        audio_token_id=5,
        audio_config=AudioEncoderConfig(d_model=128, output_dim=128, encoder_layers=1,
                                       encoder_attention_heads=2, encoder_ffn_dim=256,
                                       downsample_hidden_size=8),
        text_config=TextDecoderConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=128),
    )
    model = Qwen3ASRModel(config)
    nn.quantize(model, bits=8, group_size=32)
    mx.eval(model.parameters())
    prompts = []
    configs = [GenerationConfig(max_new_tokens=9, eos_token_ids=[63]),
               GenerationConfig(max_new_tokens=4, eos_token_ids=[63])]
    for length, audio_count in [(7, 1), (13, 3)]:
        ids = mx.array([[1] + [5] * audio_count + [2] * (length - audio_count - 1)])
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        prompts.append((ids, mx.random.normal((1, audio_count, 128)), positions))
    expected = [generate_with_info(model, *prompt, config)
                for prompt, config in zip(prompts, configs, strict=True)]
    metadata = configure_dense_prefill(model, mode)
    assert metadata["modules"] == 14  # seven linears per text layer
    assert (metadata["retained_dense_bytes"] > 0) == (mode == "cached")
    assert generate_batch(model, prompts, configs, prefill_mode="batched") == expected
    # Switching prefill policy leaves the quantized source weights and decoding
    # path intact; cached mode is not silently installing dense token generation.
    assert configure_dense_prefill(model, "off")["retained_dense_bytes"] == 0
    assert generate_batch(model, prompts, configs) == expected
