import mlx.core as mx
import pytest
from test_compiled_prefill import prompt, setup_model

from qwen_asr_opt.batch import prefill_serial
from qwen_asr_opt.chunked_joint_prefill import ChunkedJointPrefill, balanced_ranges
from qwen_asr_opt.kv_window import make_compact_step


@pytest.mark.parametrize("maximum", [1, 7, 16, 27, 64])
def test_chunked_mixed_prefix_and_ongoing_history(maximum):
    model = setup_model()
    ongoing = [prompt(7, 2, offset=3), prompt(19, 3, offset=6), prompt(9, 1)]
    fresh = prompt(27, 4, offset=7)
    tokens, keys, values = prefill_serial(model, ongoing, 64)
    original_keys = [mx.array(k) for k in keys+values]
    starts = mx.array([7, 19, 9])
    steps = len(balanced_ranges(27, maximum))
    decode = make_compact_step(model, 64)
    history = []
    dt, dk, dv = tokens, keys, values
    for n in range(steps):
        history.append(dt)
        dt, dk, dv = decode(dt, starts+n, dk, dv)
    ft, fk, fv = prefill_serial(model, [fresh], 64)
    expected = (mx.concatenate([ft, dt]),
                *[[mx.concatenate([f, d]) for f, d in zip(a, b, strict=True)] for a, b in ((fk, dk), (fv, dv))],
                mx.stack(history))
    actual = ChunkedJointPrefill(model, maximum_chunk=maximum)(fresh, tokens, starts, keys, values)
    mx.eval(actual, expected)
    assert mx.array_equal(actual[0], expected[0]).item()
    assert mx.array_equal(actual[3], expected[3]).item()
    for a, b in zip(actual[1]+actual[2], expected[1]+expected[2], strict=True):
        assert mx.allclose(a, b, atol=.02, rtol=.02).item()
    for a, b in zip(original_keys, keys+values, strict=True):
        assert mx.array_equal(a, b).item()
    for array in actual[1]+actual[2]:
        assert mx.all(array[:1, :, 27:] == 0).item()
    positions = mx.array([27, 7+steps, 19+steps, 9+steps])
    merged = make_compact_step(model, 64)
    for _ in range(4):
        actual = merged(actual[0], positions, actual[1], actual[2])
        expected = merged(expected[0], positions, expected[1], expected[2])
        assert mx.array_equal(actual[0], expected[0]).item()
        positions = positions+1


@pytest.mark.parametrize("length,maximum", [(393, 64), (393, 128), (393, 256), (7, 2)])
def test_balanced_partition_covers_once_and_respects_maximum(length, maximum):
    ranges = balanced_ranges(length, maximum)
    assert [i for start, end in ranges for i in range(start, end)] == list(range(length))
    sizes = [end-start for start, end in ranges]
    assert max(sizes) <= maximum and max(sizes)-min(sizes) <= 1
    assert len(sizes) == (length+maximum-1)//maximum
