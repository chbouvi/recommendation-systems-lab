import pandas as pd
from dashboard import get_valid_seed_movie_titles

fake_df_movies = pd.DataFrame({
    "movieId": [1, 2, 3],
    "title": ["A", "B", "C"],
    "genres": ["Adventure", "Thriller", "Action"],
})

fake_df_ratings = pd.DataFrame({
    "userId": list(range(1, 30)),
    "movieId": (
        [1] * 10 +
        [2] * 9 +
        [3] * 10
    ),
    "rating": (
        [4.0] * 10 +
        [4.0] * 9 +
        [3.5] * 10
    ),
})

def test_valid_seed_movie_titles():
    movie_titles = get_valid_seed_movie_titles(fake_df_movies, fake_df_ratings)

    assert movie_titles == ["A"]

def test_valid_seed_movie_titles_lower_min_rating():
    movie_titles = get_valid_seed_movie_titles(fake_df_movies, fake_df_ratings, min_rating=3.5)

    assert movie_titles == ["A", "C"]

def test_valid_seed_movie_titles_lower_min_rating_count():
    movie_titles = get_valid_seed_movie_titles(fake_df_movies, fake_df_ratings, min_rating_count=9)

    assert movie_titles == ["A", "B"]
