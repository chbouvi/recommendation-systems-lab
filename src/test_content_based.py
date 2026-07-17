from content_based import recommend_similar_movies

def test_valid_movie():
    movie_title = "Toy Story (1995)"

    result = recommend_similar_movies(movie_title)

    assert not result.empty

def test_original_movie_excluded():
    movie_title = "Toy Story (1995)"

    result = recommend_similar_movies(movie_title)

    assert movie_title not in result["title"].values