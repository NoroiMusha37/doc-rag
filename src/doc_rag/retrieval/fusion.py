import uuid
from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[uuid.UUID]], k: int, limit: int
) -> list[uuid.UUID]:
    """Merge several best-first rankings into one list of at most `limit` ids.

    Each id scores 1 / (k + rank) in every ranking that contains it (rank starts at 1),
    and the scores are summed, so an id found by several searches beats one found by a
    single search. Only ranks are used, which is why scores of different kinds
    (vector distance, text rank) can be combined.
    """
    scores: dict[uuid.UUID, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
    # sorted() is stable, so equal scores keep the order in which they were first seen.
    return sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
