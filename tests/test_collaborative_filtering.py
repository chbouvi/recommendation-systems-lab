import pandas as pd
from collaborative_filtering import recommend_from_similar_users

def test_valid_movie():
    movie_title = "Toy Story (1995)"

    top_movies, user_amount = recommend_from_similar_users(movie_title)
    
    assert not top_movies.empty
    assert user_amount > 0

def test_fake_movie():
    movie_title = "Fake Movie"

    top_movies, user_amount = recommend_from_similar_users(movie_title)

    assert top_movies is None
    assert user_amount is None

def test_expected_columns():
    movie_title = "Toy Story (1995)"

    top_movies, _ = recommend_from_similar_users(movie_title)

    expected_set = {
        "title",
        "collaborative_score",
        "similar_user_like_ratio",
        "similar_user_likes",
        "average_similar_user_rating",
        "genres"
    }

    top_movies_set = set(top_movies.columns)

    assert expected_set.issubset(top_movies_set)

def test_top_n():
    movie_title = "Toy Story (1995)"

    top_movies, _ = recommend_from_similar_users(movie_title, top_n=5)

    assert len(top_movies) == 5

def test_min_rating():
    movie_title = "Toy Story (1995)"

    _, user_amount1 = recommend_from_similar_users(movie_title, min_rating=4.0)
    _, user_amount2 = recommend_from_similar_users(movie_title, min_rating=4.5)

    assert user_amount2 <= user_amount1

def test_min_similar_user_likes():
    movie_title = "Toy Story (1995)"

    top_movies, _ = recommend_from_similar_users(movie_title, min_similar_user_likes=20)

    assert (top_movies["similar_user_likes"] >= 20).all()

def test_custom_dataframes_used_for_collaborative_recommendations():
    movies_df = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Seed Movie", "Popular Movie", "Other Movie"],
        "genres": ["Action", "Action", "Comedy"],
    })

    ratings_df = pd.DataFrame({
        "userId": [10, 10, 20, 20, 30],
        "movieId": [1, 2, 1, 2, 3],
        "rating": [5.0, 5.0, 5.0, 4.5, 5.0],
    })

    movie_title = "Seed Movie"

    top_movies, user_amount = recommend_from_similar_users(movie_title, ratings_df=ratings_df, movies_df=movies_df)

    assert "Popular Movie" in top_movies["title"].values
    assert "Other Movie" not in top_movies["title"].values
    assert user_amount == 2
