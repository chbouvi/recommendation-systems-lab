import pandas as pd
from dashboard import get_valid_seed_movie_titles, get_method_display_name, get_metric_winners, get_dashboard_interpretation, get_baseline_comparison

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

def test_get_dashboard_interpretation():
    sample_data = {
        "method": ["content", "collaborative", "popular", "content", "collaborative", "popular", "content", "collaborative", "popular"],
        "k": [5, 5, 5, 10, 10, 10, 20, 20, 20],
        "precision": [0.005, 0.1, 0.04, 0.06, 0.3, 0.08, 0.1, 0.2, 0.05],
        "recall": [0.005, 0.04, 0.1, 0.006, 0.01, 0.03, 0.2, 0.03, 0.09],
        "hit_rate": [0.04, 0.08, 0.006, 0.1, 0.04, 0.5, 0.1, 0.24, 0.15],
        "num_users": [100, 100, 100, 100, 100, 100, 100, 100, 100],
        "trials_per_user": [50, 50, 50, 50, 50, 50, 50, 50, 50]
    }

    sample_df = pd.DataFrame(sample_data)

    interpretation = get_dashboard_interpretation(sample_df)

    assert isinstance(interpretation, list)

    assert len(interpretation) > 0

    assert isinstance(interpretation[0], str)

    assert interpretation[0] == "Collaborative filtering has the best Hit Rate@20 at 0.2400."

    assert "Collaborative filtering beats the popular baseline" in interpretation[1]

    assert "Content-based filtering trails the popular baseline" in interpretation[2]

def test_get_baseline_comparison():
    sample_data = {
        "method": ["content", "collaborative", "popular", "content", "collaborative", "popular", "content", "collaborative", "popular"],
        "k": [5, 5, 5, 10, 10, 10, 20, 20, 20],
        "precision": [0.005, 0.1, 0.04, 0.06, 0.3, 0.08, 0.1, 0.2, 0.05],
        "recall": [0.005, 0.04, 0.1, 0.006, 0.01, 0.03, 0.2, 0.03, 0.09],
        "hit_rate": [0.04, 0.08, 0.006, 0.1, 0.04, 0.5, 0.1, 0.24, 0.15],
        "num_users": [100, 100, 100, 100, 100, 100, 100, 100, 100],
        "trials_per_user": [50, 50, 50, 50, 50, 50, 50, 50, 50]
    }

    sample_df = pd.DataFrame(sample_data)

    comparison_df = get_baseline_comparison(sample_df)

    collaborative_delta = comparison_df[(comparison_df["k"] == 5) & (comparison_df["method"] == get_method_display_name("collaborative"))][
        "precision_delta_vs_popular"
    ].values
    content_delta = comparison_df[(comparison_df["k"] == 10) & (comparison_df["method"] == get_method_display_name("content"))][
        "recall_delta_vs_popular"
    ].values

    assert len(comparison_df) == 6

    assert collaborative_delta[0] == 0.1 - 0.04

    assert content_delta[0] == 0.006 - 0.03

    assert "popular" not in comparison_df["method"].values
