import pandas as pd
from dashboard import get_valid_seed_movie_titles

df_movies = pd.read_csv("data/ml-latest-small/movies.csv")
df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

def test_valid_seed_movie_titles():
    fake_df_movies = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["A", "B", "C"],
        "genres": ["Adventure", "Thriller", "Action"],
    })

    fake_df_ratings = pd.DataFrame({
        "userId": list(range(1, 25)),
        "movieId": (
            [1] * 10 +
            [2] * 9 +
            [3] * 5
        ),
        "rating": (
            [4.0] * 10 +
            [4.0] * 9 +
            [3.5] * 5
        ),
    })

    movie_titles = get_valid_seed_movie_titles(fake_df_movies, fake_df_ratings)

    assert movie_titles == ["A"]