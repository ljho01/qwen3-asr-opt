import pytest

from qwen_asr_opt.evaluation_partition import paired_error_bootstrap, select_unseen


def test_exclusion_deduplication_and_reserved_source_separation():
    metadata = [{"id": i//3, "num_samples": 16000*(i%3+1)} for i in range(42)]
    metadata += [{"id": 100, "num_samples": 0}, {"id": 101, "num_samples": 480001}]
    selected = select_unseen(metadata, {0, 1, 2}, "Korean", evaluation=5, reserve=4)
    assert selected == select_unseen(metadata, {0, 1, 2}, "Korean", evaluation=5, reserve=4)
    used = [r["source_id"] for r in selected["evaluation"]+selected["reserve"]]
    assert len(used) == len(set(used)) == 9
    assert not set(used) & {0, 1, 2, 100, 101}
    assert not set(used) & set(selected["unused_unassigned_source_ids"])
    assert selected["eligible_source_count"] == 11 and selected["eligible_row_count"] == 33
    for row in selected["evaluation"]+selected["reserve"]:
        assert metadata[row["row_index"]]["id"] == row["source_id"]


def test_insufficient_unique_sources_does_not_spend_reserve():
    rows = [{"id": 1, "num_samples": 16000}]*20
    with pytest.raises(ValueError, match="Insufficient"):
        select_unseen(rows, set(), "English", evaluation=1, reserve=1)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_bad_partition_sizes(size):
    with pytest.raises(ValueError):
        select_unseen([], set(), "English", evaluation=size)


def test_paired_identity_and_reference_weighting():
    a = paired_error_bootstrap([(1, 100, 10, 10), (2, 1, 0, 0)], draws=30)
    assert a["difference"] == 0 and a["percentile95"] == [0, 0]
    b = paired_error_bootstrap([(1, 100, 0, 1), (2, 1, 0, 1)], draws=100)
    assert b["difference"] == 2/101  # Not the unweighted mean of utterance error rates.
    assert b == paired_error_bootstrap([(1, 100, 0, 1), (2, 1, 0, 1)], draws=100)
    c = paired_error_bootstrap([(1, 100, 1, 0), (2, 1, 1, 0)], draws=100)
    assert c["difference"] == -b["difference"]
    assert c["percentile95"][0] == pytest.approx(-b["percentile95"][1])


@pytest.mark.parametrize("rows", [[], [(1, 0, 0, 0)], [(1, 1, -1, 0)],
                                  [(1, 1, 0, 0), (1, 1, 0, 0)]])
def test_bootstrap_rejects_unpaired_or_invalid_units(rows):
    with pytest.raises(ValueError):
        paired_error_bootstrap(rows)
