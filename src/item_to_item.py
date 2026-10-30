import pandas as pd
from math import sqrt

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

def build_like_matrix(ratings_df, min_rating=4.0):
    movie_ids = sorted(ratings_df["movieId"].unique())
    user_ids = sorted(ratings_df["userId"].unique())

    liked_ratings = ratings_df[ratings_df["rating"] >= min_rating].copy()
    liked_ratings["liked"] = 1

    matrix = liked_ratings.pivot_table(
        index="movieId",
        columns="userId",
        values="liked",
        fill_value=0
    )

    matrix = matrix.reindex(
        index=movie_ids,
        columns=user_ids,
        fill_value=0
    )

    return matrix.astype(int)

def score_candidates_from_seed(matrix, seed_id):
    if seed_id not in matrix.index:
        return {}

    seed_row = matrix.loc[seed_id]

    if not seed_row.to_numpy().any():
        return {}

    seed_row_length = sqrt((seed_row ** 2).sum())
    scores = {}

    for movie_id, row in matrix.iterrows():
        if movie_id == seed_id:
            continue

        if not row.to_numpy().any():
            continue

        dot = row.dot(seed_row)

        if dot > 0:
            score = dot / (sqrt((row ** 2).sum()) * seed_row_length)
            scores[movie_id] = score

    return scores

def recommend_top_k(scores, k):
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    top_k = sorted_scores[:k]

    top_k_movies = []

    for pair in top_k:
        top_k_movies.append(pair[0])

    return top_k_movies

def recommend_item_to_item(movie_id, top_n=10, ratings_df=None, movies_df=None):
    matrix = build_like_matrix(ratings_df)

    scores = score_candidates_from_seed(matrix, movie_id)

    return recommend_top_k(scores, top_n)


if __name__ == "__main__":
    df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    print(build_like_matrix(df_ratings))
