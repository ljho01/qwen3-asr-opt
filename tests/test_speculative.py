import mlx.core as mx
import pytest
from mlx_qwen3_asr.config import AudioEncoderConfig, Qwen3ASRConfig, TextDecoderConfig
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from mlx_qwen3_asr.model import Qwen3ASRModel

from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.speculative import generate_speculative, make_block_verifier


def small_model_config():
    return Qwen3ASRConfig(
        audio_token_id=5,
        audio_config=AudioEncoderConfig(d_model=128, output_dim=128, encoder_layers=1,
                                       encoder_attention_heads=2, encoder_ffn_dim=256,
                                       downsample_hidden_size=8),
        text_config=TextDecoderConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=128,
                                     tie_word_embeddings=False),
    )


def make_prompts():
    prompts = []
    for length in (5, 9):
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        prompts.append((ids, mx.random.normal((1, 1, 128)), positions))
    return prompts


@pytest.mark.parametrize("mode", ["scatter", "rows"])
def test_block_cache_matches_stock_and_masks_rejected_future(mode):
    mx.random.seed(19)
    model = Qwen3ASRModel(small_model_config())
    model.eval()
    prompts = make_prompts()
    _, base_keys, base_values = prefill_serial(model, prompts, 32)
    lengths, common = mx.array([5, 9]), mx.array(9)
    verify = make_block_verifier(model, mode)
    tokens = mx.array([[10, 11, 12, 13], [20, 21, 22, 23]])
    block, block_keys, block_values = verify(tokens, mx.array([9, 9]), lengths,
                                            common, base_keys, base_values)
    mx.eval(block, block_keys, block_values)
    expected, stock_caches = [], []
    for row, prompt in enumerate(prompts):
        cache = model.create_cache(max_seq_len=32)
        model.prefill(*prompt, cache)
        predictions = []
        for column in range(4):
            position = prompt[0].shape[1] + column
            logits = model.step(tokens[row:row + 1, column:column + 1],
                                mx.full((1, 3, 1), position, dtype=mx.int32), cache)
            predictions.append(mx.argmax(logits, axis=-1).item())
        expected.append(predictions)
        stock_caches.append(cache)
    assert block.tolist() == expected
    for name, actual in [("keys", block_keys), ("values", block_values)]:
        for layer, array in enumerate(actual):
            for row, cache in enumerate(stock_caches):
                length = prompts[row][0].shape[1]
                assert mx.allclose(array[row, :, 9:13], getattr(cache, name)[layer][0, :, length:length + 4],
                                   atol=1e-5, rtol=1e-5).item()
    starts = mx.array([10, 12])
    correction = mx.array([[31, 32], [41, 42]])
    contaminated = verify(correction, starts, lengths, common, block_keys, block_values)
    _, clean_keys, clean_values = verify(tokens[:, :3], mx.array([9, 9]), lengths,
                                         common, base_keys, base_values)
    clean = verify(correction, starts, lengths, common, clean_keys, clean_values)
    mx.eval(contaminated, clean)
    assert contaminated[0].tolist() == clean[0].tolist()
    for a, b in zip(contaminated[1] + contaminated[2], clean[1] + clean[2], strict=True):
        for row, end in enumerate([12, 14]):
            assert mx.allclose(a[row, :, :end], b[row, :, :end], atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("same_draft,proposals", [(True, 3), (False, 1), (False, 4)])
def test_speculation_matches_target_across_acceptance_rejection_and_eos(same_draft, proposals):
    mx.random.seed(23)
    target = Qwen3ASRModel(small_model_config())
    target.eval()
    draft = target if same_draft else Qwen3ASRModel(small_model_config())
    draft.eval()
    prompts = make_prompts()
    configs = [GenerationConfig(max_new_tokens=7, eos_token_ids=[]),
               GenerationConfig(max_new_tokens=13, eos_token_ids=[])]
    expected = [generate_with_info(target, *p, c) for p, c in zip(prompts, configs, strict=True)]
    actual, stats = generate_speculative(target, draft, prompts, prompts, configs, proposals)
    assert actual == expected
    if same_draft:
        assert stats["accepted_proposals"] > 0
    else:
        assert stats["accepted_proposals"] < stats["proposed_tokens"]
    configs[0] = GenerationConfig(max_new_tokens=7, eos_token_ids=[expected[0].tokens[0]])
    actual, _ = generate_speculative(target, draft, prompts, prompts, configs, proposals)
    assert actual[0].finish_reason == "eos" and actual[0].tokens == []
    assert actual[1] == expected[1]
