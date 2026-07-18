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

