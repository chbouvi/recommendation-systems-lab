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

if __name__ == "__main__":
    relevant_ids = get_relevant_movies_for_user(1)
    training_ids, hidden_ids = split_relevant_movies(relevant_ids)
    seed_movie = random.choice(training_ids)
    seed_title = get_movie_title(seed_movie)
    k = 5

    recommendations = recommend_similar_movies(seed_title, top_n=k)
    recommended_ids = recommendations["movieId"].to_list()
    recommended_titles = recommendations["title"].to_list()

    print(f"Seed movie: {seed_title}")
    print(f"Hidden movie: {get_movie_title(hidden_ids[0])}")
    print(f"Top {k} Recommendations: {recommended_titles}")
    print(f"Precision@{k}: {precision_at_k(recommended_ids, hidden_ids, k)}")
    print(f"Recall@{k}: {recall_at_k(recommended_ids, hidden_ids, k)}")
    print(f"Hit Rate@{k}: {hit_rate_at_k(recommended_ids, hidden_ids, k)}")


    