import mlx.core as mx
import pytest
from test_compiled_prefill import prompt, setup_model

from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.joint_prefill import JointPrefillStep
from qwen_asr_opt.kv_window import make_compact_step


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [1, 3, 7])
def test_joint_prefill_isolates_recordings_and_retains_cache(compiled, rows):
    model = setup_model()
    ongoing = [prompt(7 + 2*i, i % 3, offset=i+2) for i in range(rows)]
    fresh = prompt(27, 3, offset=4)
    tokens, keys, values = prefill_serial(model, ongoing, 64)
    positions = mx.array([p[0].shape[1] for p in ongoing])
    step = make_compact_step(model, 64)
    # Existing rows have generated history as well as audio prefixes.
    for _ in range(3):
        tokens, keys, values = step(tokens, positions, keys, values)
        positions = positions + 1
    mx.eval(tokens, keys, values)
    before = [mx.array(v) for v in keys + values]
    expected_fresh = prefill_serial(model, [fresh], 64)
    expected_decode = step(tokens, positions, keys, values)
    expected = (mx.concatenate([expected_fresh[0], expected_decode[0]], axis=0),
                *[[mx.concatenate([f, d], axis=0) for f, d in zip(expected_fresh[i], expected_decode[i], strict=True)]
                  for i in (1, 2)])
    actual = JointPrefillStep(model, compiled=compiled)(fresh, tokens, positions, keys, values)
    mx.eval(actual, expected)
    assert mx.array_equal(actual[0], expected[0]).item()
    for a, b in zip(actual[1] + actual[2], expected[1] + expected[2], strict=True):
        assert mx.allclose(a, b, atol=.01, rtol=.01).item()
    for a, b in zip(before, keys+values, strict=True):
        assert mx.array_equal(a, b).item()
    for array in actual[1] + actual[2]:
        assert mx.all(array[:1, :, 27:] == 0).item()
    # Continuation uses the same ordinary compiled token decoder after the merge.
    starts = mx.concatenate([mx.array([27]), positions+1])
    merged_step = make_compact_step(model, 64)
    for _ in range(4):
        actual = merged_step(actual[0], starts, actual[1], actual[2])
        expected = merged_step(expected[0], starts, expected[1], expected[2])
        assert mx.array_equal(actual[0], expected[0]).item()
        starts = starts+1


def test_joint_prefill_keeps_upstream_input_validation():
    model = setup_model()
    p = prompt(7, 2)
    tokens, keys, values = prefill_serial(model, [p], 16)
    step = JointPrefillStep(model)
    with pytest.raises(ValueError):
        step(prompt(17, 2), tokens, mx.array([7]), keys, values)
    with pytest.raises((ValueError, TypeError)):
        step((p[0], p[1][:, :1], p[2]), tokens, mx.array([7]), keys, values)
    with pytest.raises(ValueError):
        step(p, tokens, mx.array([7, 8]), keys, values)
