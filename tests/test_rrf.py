from app.rag import reciprocal_rank_fusion


def test_rrf_rewards_results_present_in_both_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)

    assert fused[0][0] == "b"
    assert fused[0][1] > fused[1][1]


def test_rrf_does_not_duplicate_result_ids() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"], ["a"]], k=60)

    assert [item_id for item_id, _ in fused].count("a") == 1
