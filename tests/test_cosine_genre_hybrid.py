import pytest
import pandas as pd
from cosine_genre_hybrid import get_cosine_genre_scores, get_genre_similarity_scores
from math import sqrt

def test_genre_similarity_scores():
    movies_df = pd.DataFrame({
        "movieId": [10, 20, 30],
        "genres": ["Drama|Romance", "Documentary|Drama", "Action"],
    })

    genre_scores = get_genre_similarity_scores(10, [10, 20, 30], movies_df)

    assert genre_scores[10] == 1
    assert genre_scores[20] == 1/3
    assert genre_scores[30] == 0

def test_no_shared_fans_candidate():
    movies_df = pd.DataFrame({
        "movieId": [10, 20, 30],
        "genres": ["Drama|Romance", "Documentary|Drama", "Action"],
    })

    matrix = pd.DataFrame(
        [
            [1, 1, 0],  # movie 10: seed
            [1, 0, 0],  # movie 20: one shared liker
            [0, 0, 1],  # movie 30: no shared likers
        ],
        index=[10, 20, 30],
        columns=[1, 2, 3],
    )

    hybrid_scores = get_cosine_genre_scores(matrix, 10, movies_df)

    assert 10 not in hybrid_scores
    assert hybrid_scores[20] == pytest.approx(0.8 * (1 / sqrt(2)) + 0.2 * (1 / 3))
    assert 30 not in hybrid_scores
