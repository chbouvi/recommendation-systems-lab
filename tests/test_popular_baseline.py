from popular_baseline import recommend_popular_movies

def test_valid_movie():
    movie_title = "Toy Story (1995)"
    result = recommend_popular_movies(movie_title)

    assert not result.empty

def test_fake_movie():
    movie_title = "Fake Movie"
    result = recommend_popular_movies(movie_title)

    assert result is None

def test_original_movie_excluded():
    movie_title = "Toy Story (1995)"

    result = recommend_popular_movies(movie_title)

    assert movie_title not in result["title"].values

def test_expected_columns():
    movie_title = "Toy Story (1995)"

    result = recommend_popular_movies(movie_title)

    expected_set = {
        "movieId",
        "high_rating_count",
        "average_rating",
        "title",
        "genres"
    }

    result_set = set(result.columns)

    assert expected_set.issubset(result_set)

def test_top_n():
    movie_title = "Toy Story (1995)"

    result = recommend_popular_movies(movie_title, top_n=5)

    assert len(result) == 5
