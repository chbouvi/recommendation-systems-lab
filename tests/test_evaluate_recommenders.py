from evaluate_recommenders import get_relevant_movies_for_user, split_relevant_movies, run_evaluation_for_k_values, run_evaluation_for_users

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

    assert len(training_ids) == 3
    assert len(hidden_ids) == 1
    assert hidden_ids[0] not in training_ids
    assert len(training_ids) + len(hidden_ids) == len(relevant_movie_ids)

def test_split_relevant_movies_empty():
    training_ids, hidden_ids = split_relevant_movies([])

    assert training_ids == []
    assert hidden_ids == []

def test_run_evaluation_for_k_values():
    user_id = 1
    k_values = [5, 10, 20]
    num_trials = 5

    average_scores, completed_trials = run_evaluation_for_k_values(user_id, k_values, num_trials)

    assert completed_trials > 0
    assert 5 in average_scores
    assert 10 in average_scores
    assert 20 in average_scores
    
    for k in k_values:
        assert "precision" in average_scores[k]
        assert "recall" in average_scores[k]
        assert "hit_rate" in average_scores[k]
        assert (average_scores[k]["precision"] >= 0) and (average_scores[k]["precision"] <= 1)
        assert (average_scores[k]["recall"] >= 0) and (average_scores[k]["recall"] <= 1)
        assert (average_scores[k]["hit_rate"] >= 0) and (average_scores[k]["hit_rate"] <= 1)

def test_run_evaluation_for_users():
    user_ids = [1, 2]
    k_values = [5, 10]
    num_trials = 5

    average_scores_by_k = run_evaluation_for_users(user_ids, k_values, num_trials)

    assert 5 in average_scores_by_k 
    assert 10 in average_scores_by_k

    for k in k_values:
        assert "precision" in average_scores_by_k[k]
        assert "recall" in average_scores_by_k[k]
        assert "hit_rate" in average_scores_by_k[k]
        assert (average_scores_by_k[k]["precision"] >= 0) and (average_scores_by_k[k]["precision"] <= 1)
        assert (average_scores_by_k[k]["recall"] >= 0) and (average_scores_by_k[k]["recall"] <= 1)
        assert (average_scores_by_k[k]["hit_rate"] >= 0) and (average_scores_by_k[k]["hit_rate"] <= 1)
