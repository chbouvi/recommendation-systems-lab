import pandas as pd

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

def get_relevant_movies_for_user(user_id, min_rating=4.0):
    ratings_for_user = df_ratings[
        (df_ratings["userId"] == user_id) &
        (df_ratings["rating"] >= min_rating)
    ]

    return ratings_for_user["movieId"].to_list()

def split_relevant_movies(relevant_movie_ids):
    if not relevant_movie_ids:
        return [], []
    
    training_ids = relevant_movie_ids[:-1]
    hidden_ids = [relevant_movie_ids[-1]]

    return training_ids, hidden_ids


if __name__ == "__main__":
    result = get_relevant_movies_for_user(1)

    print(f"User 1 liked {len(result)} movies")
    print(result)
    print()

    training_ids, hidden_ids = split_relevant_movies(result)

    print(f"Training movies: {training_ids}")
    print()
    print(f"Hidden movie: {hidden_ids}")
    