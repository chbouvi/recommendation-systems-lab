import pandas as pd
from item_to_item import build_like_matrix

def test_build_like_matrix():
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [10, 20, 10, 30, 20, 30],
        "rating": [5.0, 2.0, 4.0, 4.5, 2.0, 3.0],
    })

    matrix = build_like_matrix(ratings_df)

    assert matrix.loc[10, 1] == 1
    assert matrix.loc[10, 3] == 0
    assert matrix.loc[20, 2] == 0
    assert matrix.loc[30, 2] == 1
