from item_to_item import build_like_matrix, score_candidates_from_seed, recommend_top_k
import pandas as pd

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

def get_scaled_popularity_scores(matrix):
    like_counts = matrix.sum(axis=1)

    if len(like_counts) == 0:
        return {}

    largest_count = max(like_counts)

    if largest_count == 0:
        return like_counts.to_dict()

    like_counts = like_counts / largest_count
    return like_counts.to_dict()

def get_hybrid_scores(matrix, seed_id, cosine_weight=0.8):
    cosine_scores = score_candidates_from_seed(matrix, seed_id)
    popularity_scores = get_scaled_popularity_scores(matrix)

    hybrid_scores = {}

    for movie_id, popularity_score in popularity_scores.items():
        if movie_id == seed_id:
            continue
        cosine_score = cosine_scores.get(movie_id, 0)
        hybrid_score = cosine_weight * cosine_score + (1 - cosine_weight) * popularity_score
        hybrid_scores[movie_id] = hybrid_score

    return hybrid_scores

def recommend_hybrid(seed_id, top_n, ratings_df, cosine_weight=0.8):
    matrix = build_like_matrix(ratings_df)
    scores = get_hybrid_scores(matrix, seed_id, cosine_weight)

    return recommend_top_k(scores, top_n)


if __name__ == "__main__":
    matrix = build_like_matrix(df_ratings)
    result = get_hybrid_scores(matrix, 1)
    print(result)
