import pytest
import pandas as pd
from hybrid import get_hybrid_scores
from math import sqrt

def test_no_shared_fans_candidate():
    matrix = pd.DataFrame(
    [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    index=[10, 20, 30],
    columns=[1, 2, 3],
    )

    hybrid_scores = get_hybrid_scores(matrix, 10)

    assert 10 not in hybrid_scores
    assert hybrid_scores[30] == pytest.approx(0.1)
    assert hybrid_scores[20] == pytest.approx(0.8 * (1 / sqrt(2)) + 0.2 * 0.5)