import pandas as pd
from dashboard import get_valid_seed_movie_titles, get_method_display_name, get_metric_winners

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

def test_get_valid_method_display_name_known_method():
    method = "content"

    best_method = get_method_display_name(method)

    assert best_method == "Content-based filtering"

def test_get_method_display_name_unknown_method():
    method = "other"

    best_method = get_method_display_name(method)

    assert best_method == "other"

def test_get_metric_winners():
    sample_data = {
        "method": ["content", "collaborative", "content", "collaborative"],
        "k": [5, 5, 10, 10],
        "precision": [0.005, 0.1, 0.04, 0.06],
        "recall": [0.005, 0.04, 0.1, 0.006],
        "hit_rate": [0.04, 0.08, 0.006, 0.1],
        "num_users": [100, 100, 100, 100],
        "trials_per_user": [50, 50, 50, 50]
    }

    sample_df = pd.DataFrame(sample_data)

    metric_winners = get_metric_winners(sample_df)

    assert len(metric_winners) == 6

    precision_k5 = metric_winners[
        (metric_winners["k"] == 5) &
        (metric_winners["metric"] == "precision")
    ].iloc[0]

    recall_k10 = metric_winners[
            (metric_winners["k"] == 10) &
            (metric_winners["metric"] == "recall")
    ].iloc[0]

    hit_rate_k10 = metric_winners[
            (metric_winners["k"] == 10) &
            (metric_winners["metric"] == "hit_rate")
    ].iloc[0]

    assert precision_k5["best_method"] == "Collaborative filtering"

    assert precision_k5["best_value"] == 0.1

    assert recall_k10["best_method"] == "Content-based filtering"

    assert recall_k10["best_value"] == 0.1

    assert hit_rate_k10["best_method"] == "Collaborative filtering"

    assert hit_rate_k10["best_value"] == 0.1
