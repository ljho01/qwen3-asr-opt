import mlx.core as mx
import pytest
from mlx_qwen3_asr.generate import GenerationConfig, generate_with_info
from mlx_qwen3_asr.model import Qwen3ASRModel
from test_speculative import small_model_config

from qwen_asr_opt.continuous import generate_continuous


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("kv_policy", ["padded", "compact", "compact128", "growing128",
                                      "adaptive128", "adaptive64"])
def test_continuous_refill_matches_independent_target_and_preserves_input_order(batch_size, kv_policy):
    mx.random.seed(37)
    model = Qwen3ASRModel(small_model_config())
    model.eval()
    jobs = []
    for length, limit in zip([5, 9, 7, 4, 11, 3] * 2, [2, 13, 5, 1, 9, 7] * 2, strict=True):
        ids = mx.array([[1, 5] + [2] * (length - 2)])
        position = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        jobs.append(((ids, mx.random.normal((1, 1, 128)), position),
                     GenerationConfig(max_new_tokens=limit, eos_token_ids=[])))
    expected = [generate_with_info(model, *p, c) for p, c in jobs]
    supplied = []
    completed = []

    def stream():
        for index, job in enumerate(jobs):
            supplied.append(index)
            yield job

    actual, stats = generate_continuous(
        model, stream(), batch_size, shared_prefix=16, token_budget=16, kv_policy=kv_policy,
        on_complete=lambda index, result: completed.append((index, result, len(supplied))),
    )
    assert actual == expected
    assert sorted((index, result) for index, result, _ in completed) == list(enumerate(expected))
    # Persist the first completion before requesting the next audio chunk.
    assert completed[0][2] == batch_size
    if batch_size > 1:
        assert [index for index, _, _ in completed] != list(range(len(jobs)))
    assert stats["prefills"] == len(jobs)
    assert stats["refills"] == len(jobs) - batch_size
    assert stats["peak_active_rows"] <= batch_size
    # A row that immediately emits EOS must be replaced without contaminating its successor.
    prompt, _ = jobs[1]
    jobs[1] = (prompt, GenerationConfig(max_new_tokens=13, eos_token_ids=[expected[1].tokens[0]]))
    expected[1] = generate_with_info(model, *jobs[1][0], jobs[1][1])
    actual, _ = generate_continuous(model, iter(jobs), batch_size,
                                     shared_prefix=16, token_budget=16, kv_policy=kv_policy)
    assert actual == expected


def test_empty_continuous_pool_and_rejected_capacity():
    model = Qwen3ASRModel(small_model_config())
    result, stats = generate_continuous(model, [], 4)
    assert result == [] and stats["decode_calls"] == 0
    ids = mx.array([[1, 5, 2]])
    positions = mx.broadcast_to(mx.arange(3)[None, None, :], (1, 3, 3))
    job = ((ids, mx.zeros((1, 1, 128)), positions), GenerationConfig(max_new_tokens=1))
    with pytest.raises(ValueError, match="Prompt length"):
        generate_continuous(model, [job], shared_prefix=2)
