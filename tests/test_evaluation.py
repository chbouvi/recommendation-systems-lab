from evaluation import precision_at_k

def test_precision_at_k():
    recommended_ids = [1, 2, 3, 4, 5, 6]
    relevant_ids = [2, 4, 8]
    k = 5

    result = precision_at_k(recommended_ids, relevant_ids, k)

    assert result == 0.4

