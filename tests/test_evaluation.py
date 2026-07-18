import pytest
from evaluation import precision_at_k, recall_at_k, hit_rate_at_k

def test_valid_precision_at_k():
    recommended_ids = [1, 2, 3, 4, 5, 6]
    relevant_ids = [2, 4, 8]
    k = 5

    result = precision_at_k(recommended_ids, relevant_ids, k)

    assert result == 0.4

def test_precision_at_k_zero_k():
    recommended_ids = [1, 2, 3, 4, 5, 6]
    relevant_ids = [2, 4, 8]
    k = 0

    result = precision_at_k(recommended_ids, relevant_ids, k)

    assert result == 0

def test_valid_recall_at_k():
    recommended_ids = [1, 2, 3, 4, 5, 6]
    relevant_ids = [2, 4, 8]
    k = 5

    result = recall_at_k(recommended_ids, relevant_ids, k)

    assert result == pytest.approx(2/3)

def test_recall_at_k_empty_relevant_ids():
    recommended_ids = [1, 2, 3, 4, 5, 6]
    relevant_ids = []
    k = 5

    result = recall_at_k(recommended_ids, relevant_ids, k)

    assert result == 0

def test_hit_rate_at_k_hit():
    recommended_ids = [1, 2, 3, 4, 5, 6]
    relevant_ids = [9, 10, 2]
    k = 5

    result = hit_rate_at_k(recommended_ids, relevant_ids, k)

    assert result == 1

def test_hit_rate_at_k_no_hit():
    recommended_ids = [1, 2, 3, 4, 5, 6]
    relevant_ids = [9, 10, 11]
    k = 5

    result = hit_rate_at_k(recommended_ids, relevant_ids, k)

    assert result == 0