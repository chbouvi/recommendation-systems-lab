from evaluate_recommenders import get_relevant_movies_for_user, split_relevant_movies

def test_get_relevant_movies_for_user_valid():
    user_id = 1
    
    result = get_relevant_movies_for_user(user_id)

    assert len(result) >= 1

def test_get_relevant_movies_for_user_invalid():
    user_id = 123456789

    result = get_relevant_movies_for_user(user_id)

    assert result == []

def test_split_relevant_movies_valid():
    relevant_movie_ids = [5, 10, 15, 20]

    training_ids, hidden_ids = split_relevant_movies(relevant_movie_ids)

    assert training_ids == [5, 10, 15]
    assert hidden_ids == [20]

def test_split_relevant_movies_empty():
    training_ids, hidden_ids = split_relevant_movies([])

    assert training_ids == []
    assert hidden_ids == []