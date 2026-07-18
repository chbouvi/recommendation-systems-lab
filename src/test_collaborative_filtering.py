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