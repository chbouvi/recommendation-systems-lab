from item_to_item import build_like_matrix, score_candidates_from_seed, recommend_top_k
import pandas as pd

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
df_movies = pd.read_csv("data/ml-latest-small/movies.csv")

def get_genre_similarity_scores(seed_id, candidate_ids, movies_df):
    genre_matrix = movies_df.set_index("movieId")["genres"].str.get_dummies(sep="|")
    seed_movie = genre_matrix.loc[seed_id]
    similarity_scores = {}

    for candidate_id in candidate_ids:
        candidate_movie = genre_matrix.loc[candidate_id]
        intersection = sum(seed_movie & candidate_movie)
        union = sum(seed_movie | candidate_movie)
        similarity_scores[candidate_id] = intersection / union

    return similarity_scores 

def get_cosine_genre_scores(matrix, seed_id, movies_df=df_movies, cosine_weight=0.8):
    cosine_scores = score_candidates_from_seed(matrix, seed_id)
    genre_scores = get_genre_similarity_scores(seed_id, cosine_scores.keys(), movies_df)

    hybrid_scores = {}

    for movie_id, genre_score in genre_scores.items():
        if movie_id == seed_id:
            continue
        cosine_score = cosine_scores.get(movie_id, 0)
        hybrid_score = cosine_weight * cosine_score + (1 - cosine_weight) * genre_score
        hybrid_scores[movie_id] = hybrid_score

    return hybrid_scores

def recommend_cosine_genre_hybrid(seed_id, top_n, ratings_df, cosine_weight=0.8):
    matrix = build_like_matrix(ratings_df)
    scores = get_cosine_genre_scores(matrix, seed_id, cosine_weight=cosine_weight)

    return recommend_top_k(scores, top_n)


if __name__ == "__main__":
    matrix = build_like_matrix(df_ratings)
    result = get_cosine_genre_scores(matrix, 1)
    print(result)
