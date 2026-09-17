"""Deterministic matched-pair sampling shared by vector and decoder objectives."""

from __future__ import annotations

import random
from typing import Sequence, TypeVar


Record = TypeVar("Record")


def sample_paired_family(records: Sequence[Record], rng: random.Random, pairs: int) -> list[Record]:
    """Choose one family then sample matched A/B records with replacement.

    Keeping the order and RNG calls identical lets independently run objective forks
    consume exactly the same pair sequence when their seed and data are identical.
    """
    if pairs < 1:
        raise ValueError("pairs must be positive")
    families = tuple(sorted({record.family_id for record in records}))
    if not families:
        raise ValueError("no records")
    family = rng.choice(families)
    candidates = [record for record in records if record.family_id == family]
    if not candidates:
        raise RuntimeError("chosen family has no records")
    return [candidates[rng.randrange(len(candidates))] for _ in range(pairs)]
