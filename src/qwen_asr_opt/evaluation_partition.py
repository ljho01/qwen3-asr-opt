"""Deterministic source-utterance partitions and paired descriptive uncertainty."""
from __future__ import annotations

import hashlib
import math
import random


def select_unseen(metadata, excluded, language, *, evaluation=60, reserve=30,
                  salt="mlp6-expanded-20260917"):
    """Choose one <=30s recording per unused source ID before seeing its text/audio."""
    if any(type(n) is not int or n < 1 for n in (evaluation, reserve)):
        raise ValueError("Positive evaluation and reserve sizes required")
    groups = {}
    for index, row in enumerate(metadata):
        identity, samples = row["id"], row["num_samples"]
        if type(identity) is not int or type(samples) is not int:
            raise ValueError("Integer source IDs and sample counts required")
        if identity in excluded or not 0 < samples <= 30*16000:
            continue
        groups.setdefault(identity, []).append({"row_index": index, "source_id": identity,
                                                "num_samples": samples})

    def key(kind, identity):
        return hashlib.sha256(f"{salt}:{language}:{kind}:{identity}".encode()).digest()

    identities = sorted(groups, key=lambda identity: (key("source", identity), identity))
    if len(identities) < evaluation+reserve:
        raise ValueError("Insufficient unused source utterances for both disjoint partitions")
    chosen = [min(groups[identity], key=lambda row: (key("row", row["row_index"]), row["row_index"]))
              for identity in identities[:evaluation+reserve]]
    return {"evaluation": chosen[:evaluation], "reserve": chosen[evaluation:],
            "unused_unassigned_source_ids": identities[evaluation+reserve:],
            "eligible_source_count": len(identities),
            "eligible_row_count": sum(len(rows) for rows in groups.values())}


def paired_error_bootstrap(rows, *, draws=10000, seed=20260917):
    """Resample matched unique utterances; ratios use total errors/total reference.

    rows = (source_id, reference_units, baseline_errors, candidate_errors).
    Return candidate-minus-baseline error-rate differences (fractions, not percent).
    Percentile intervals are descriptive, not proof of equivalence or significance.
    """
    if type(draws) is not int or draws < 1 or not rows:
        raise ValueError("Nonempty paired rows and positive integer draws required")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Require one paired observation per source utterance")
    if any(len(r) != 4 or any(type(x) is not int for x in r[1:])
           or r[1] <= 0 or min(r[2:]) < 0 for r in rows):
        raise ValueError("Positive integer reference counts and nonnegative error counts required")
    n = len(rows)
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        units = delta = 0
        for _ in range(n):
            _, size, before, after = rows[rng.randrange(n)]
            units += size
            delta += after-before
        differences.append(delta/units)
    differences.sort()

    def quantile(p):
        index = (draws-1)*p
        lo = math.floor(index)
        hi = math.ceil(index)
        return differences[lo] + (index-lo)*(differences[hi]-differences[lo])

    units = sum(r[1] for r in rows)
    return {"source_utterances": n, "draws": draws, "seed": seed,
            "difference": sum(r[3]-r[2] for r in rows)/units,
            "percentile95": [quantile(.025), quantile(.975)],
            "scope": "Paired unique-utterance percentile bootstrap, descriptive only; no formal equivalence claim"}
