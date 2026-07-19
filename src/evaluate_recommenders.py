import pandas as pd

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

def get_relevant_movies_for_user(user_id, min_rating=4.0):
    ratings_for_user = df_ratings[
        (df_ratings["userId"] == user_id) &
        (df_ratings["rating"] >= min_rating)
    ]

    return ratings_for_user["movieId"].to_list()


if __name__ == "__main__":
    result = get_relevant_movies_for_user(1)

    print(f"User 1 liked {len(result)} movies")
    print(result)