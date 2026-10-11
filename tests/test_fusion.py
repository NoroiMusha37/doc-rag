import uuid

import pytest

from src.doc_rag.retrieval.fusion import reciprocal_rank_fusion

A, B, C, D = (uuid.uuid4() for _ in range(4))


def test_an_item_found_by_both_searches_beats_one_found_by_a_single_search() -> None:
    fused = reciprocal_rank_fusion([[A, B, C], [C, D]], k=60, limit=10)

    # C is 3rd and 1st; A is only 1st in one list.
    assert fused[0] == C
    assert set(fused) == {A, B, C, D}


def test_the_better_rank_wins_within_a_single_ranking() -> None:
    assert reciprocal_rank_fusion([[A, B, C]], k=60, limit=10) == [A, B, C]


def test_equal_scores_keep_the_order_in_which_items_were_first_seen() -> None:
    # A and C are both first in their list, B and D both second.
    assert reciprocal_rank_fusion([[A, B], [C, D]], k=60, limit=10) == [A, C, B, D]


def test_limit_keeps_only_the_best_items() -> None:
    assert reciprocal_rank_fusion([[A, B, C, D]], k=60, limit=2) == [A, B]


@pytest.mark.parametrize("rankings", [[], [[]], [[], []]])
def test_nothing_to_fuse_gives_an_empty_list(rankings: list[list[uuid.UUID]]) -> None:
    assert reciprocal_rank_fusion(rankings, k=60, limit=10) == []


def test_one_empty_ranking_leaves_the_other_untouched() -> None:
    assert reciprocal_rank_fusion([[], [B, A]], k=60, limit=10) == [B, A]
