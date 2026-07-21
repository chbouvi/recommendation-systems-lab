import pandas as pd
import random
from content_based import recommend_similar_movies
from evaluation import precision_at_k, recall_at_k, hit_rate_at_k

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
df_movies = pd.read_csv("data/ml-latest-small/movies.csv")

def get_relevant_movies_for_user(user_id, min_rating=4.0):
    ratings_for_user = df_ratings[
        (df_ratings["userId"] == user_id) &
        (df_ratings["rating"] >= min_rating)
    ]

    return ratings_for_user["movieId"].to_list()

def split_relevant_movies(relevant_movie_ids):
    if not relevant_movie_ids:
        return [], []
    
    training_ids = relevant_movie_ids.copy()  
    hidden_ids = [random.choice(relevant_movie_ids)]
    training_ids.remove(hidden_ids[0])

    return training_ids, hidden_ids

def get_movie_title(movie_id):
    movie_row = df_movies[df_movies["movieId"] == movie_id]

    movie_title = movie_row["title"].iloc[0]

    return movie_title

def run_evaluation(user_id, k, num_trials):
    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    completed_trials = 0

    for _ in range(num_trials):
        relevant_ids = get_relevant_movies_for_user(user_id)
        training_ids, hidden_ids = split_relevant_movies(relevant_ids)

        if not training_ids:
            continue

        seed_movie = random.choice(training_ids)
        seed_title = get_movie_title(seed_movie)

        recommendations = recommend_similar_movies(seed_title, top_n=k)
        recommended_ids = recommendations["movieId"].to_list()

        precision_scores.append(precision_at_k(recommended_ids, hidden_ids, k))
        recall_scores.append(recall_at_k(recommended_ids, hidden_ids, k))
        hit_rate_scores.append(hit_rate_at_k(recommended_ids, hidden_ids, k))
        completed_trials += 1
    
    if completed_trials == 0:
        return 0, 0, 0, 0
    
    average_precision_score = sum(precision_scores) / completed_trials
    average_recall_score = sum(recall_scores) / completed_trials
    average_hit_rate_score = sum(hit_rate_scores) / completed_trials

    return average_precision_score, average_recall_score, average_hit_rate_score, completed_trials

def run_evaluation_for_k_values(user_id, k_values, num_trials):
    scores = {}

    for k in k_values:
        scores[k] = {
            "precision": [],
            "recall": [],
            "hit_rate": []
        }

    completed_trials = 0

    for _ in range(num_trials):
        relevant_ids = get_relevant_movies_for_user(user_id)
        training_ids, hidden_ids = split_relevant_movies(relevant_ids)

        if not training_ids:
            continue

        seed_movie = random.choice(training_ids)
        seed_title = get_movie_title(seed_movie)

        recommendations = recommend_similar_movies(seed_title, top_n=max(k_values))
        recommended_ids = recommendations["movieId"].to_list()

        for k in k_values:
            precision = precision_at_k(recommended_ids, hidden_ids, k)
            recall = recall_at_k(recommended_ids, hidden_ids, k)
            hit_rate = hit_rate_at_k(recommended_ids, hidden_ids, k)

            scores[k]["precision"].append(precision)
            scores[k]["recall"].append(recall)
            scores[k]["hit_rate"].append(hit_rate)

        
        completed_trials += 1

    if completed_trials == 0:
        return {}, 0
    
    average_scores = {}

    for k in k_values:
        average_precision = sum(scores[k]["precision"]) / completed_trials
        average_recall = sum(scores[k]["recall"]) / completed_trials
        average_hit_rate = sum(scores[k]["hit_rate"]) / completed_trials

        average_scores[k] = {
            "precision": average_precision,
            "recall": average_recall,
            "hit_rate": average_hit_rate
        }
    
    return average_scores, completed_trials

        

if __name__ == "__main__":
    user_id = 1
    k_values = [5, 10, 20]
    num_trials = 50

    average_scores, completed_trials = run_evaluation_for_k_values(user_id, k_values, num_trials)

    print(f"Completed trials: {completed_trials}/{num_trials}")
    print()
    
    for k in average_scores:
        print(f"Average precision score@{k}: {average_scores[k]['precision']}")
        print(f"Average recall score@{k}: {average_scores[k]['recall']}")
        print(f"Average hit rate score@{k}: {average_scores[k]['hit_rate']}")
        print()
    