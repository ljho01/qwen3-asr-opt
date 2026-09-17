import mlx.core as mx
import pytest
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from test_compiled_prefill import prompt, setup_model

from qwen_asr_opt import continuous
from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.compiled_prefill import prefix_outputs
from qwen_asr_opt.terminal_prefill import TerminalPrefill, terminal_prefix_outputs


@pytest.mark.parametrize("length", [1, 7, 31, 128, 257])
@pytest.mark.parametrize("compiled", [False, True])
def test_final_logits_and_all_prefix_kv_match_full_last_layer(length, compiled):
    model = setup_model()
    ids, features, positions = prompt(length, 0 if length == 1 else 2, offset=7)
    # Unequal temporal/height/width coordinates catch accidentally using one axis.
    positions = positions + mx.array([0, 11, 23])[None, :, None]
    embeds = model._inject_audio_features(model._embed_tokens(ids), features, ids == model.audio_token_id)
    expected = prefix_outputs(model, embeds, positions)
    function = lambda x, p: terminal_prefix_outputs(model, x, p)
    actual = (mx.compile(function) if compiled else function)(embeds, positions)
    assert actual[0].shape == (1, 1, model.config.text_config.vocab_size)
    assert mx.allclose(actual[0], expected[0], atol=.006, rtol=.006).item()
    assert mx.array_equal(mx.argmax(actual[0], -1), mx.argmax(expected[0], -1)).item()
    for a, b in zip(actual[1]+actual[2], expected[1]+expected[2], strict=True):
        assert a.shape[2] == length and mx.allclose(a, b, atol=.004, rtol=.004).item()


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("compiled", [False, True])
def test_refills_caps_and_continuation_keep_independent_streams(monkeypatch, batch_size, compiled):
    model = setup_model()
    jobs = [(prompt(length, count), GenerationConfig(max_new_tokens=cap, eos_token_ids=[]))
            for length, count, cap in [(7, 1, 5), (19, 3, 2), (131, 7, 7), (69, 0, 4), (17, 2, 6)]]
    expected = [generate_with_info(model, *value, config) for value, config in jobs]
    helper = TerminalPrefill(model, compiled=compiled, max_graphs=2)
    monkeypatch.setattr(continuous, "prefill_serial", helper)
    actual, stats = continuous.generate_continuous(model, jobs, batch_size, kv_policy="growing128")
    assert actual == expected and stats["refills"] == max(0, len(jobs)-batch_size)
    assert helper.calls == len(jobs) and len(helper.graphs) <= 2


def test_packed_cache_padding_lru_and_validation():
    model = setup_model()
    helper = TerminalPrefill(model, max_graphs=2)
    for lengths in [(7, 17), (69, 129), (7, 17), (7, 17)]:
        prompts = [prompt(length, i+1) for i, length in enumerate(lengths)]
        reference = prefill_serial(model, prompts, 256)
        actual = helper(model, prompts, 256)
        assert mx.array_equal(actual[0], reference[0]).item()
        for a, b in zip(actual[1]+actual[2], reference[1]+reference[2], strict=True):
            assert mx.allclose(a, b, atol=.004, rtol=.004).item()
            for row, length in enumerate(lengths):
                assert mx.all(a[row, :, length:] == 0).item()
    assert helper.graphs_created == 6 and helper.evictions == 4 and helper.cache_hits == 2
    calls = helper.calls
    ids, features, positions = prompt(7, 2)
    for invalid in [(ids, features[:, :1], positions), (ids*99999, features, positions),
                    (ids.astype(mx.float32), features, positions)]:
        with pytest.raises((ValueError, TypeError)):
            helper(model, [invalid], 128)
    assert helper.calls == calls
    with pytest.raises(ValueError):
        helper(setup_model(), [(ids, features, positions)], 128)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_cache_bound(count):
    with pytest.raises(ValueError):
        TerminalPrefill(setup_model(), max_graphs=count)
