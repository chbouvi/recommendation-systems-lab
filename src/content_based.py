import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)

df_movies = pd.read_csv("data/ml-latest-small/movies.csv")
df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

rating_stats = (
    df_ratings.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .reset_index()
)

rating_stats.columns = ["movieId", "average_rating", "rating_count"]

def recommend_similar_movies(movie_title, top_n=10):
    movie_row = df_movies[df_movies["title"] == movie_title]

    if movie_row.empty:
        print("Movie not found.")
        return
    
    movie_index = movie_row.index[0]
    
    genre_matrix = df_movies["genres"].str.get_dummies(sep="|")

    selected_movie = genre_matrix.loc[movie_index]

    similarity_scores = genre_matrix.dot(selected_movie)

    similarity_results = df_movies[["movieId", "title", "genres"]].copy()
    similarity_results["similarity_score"] = similarity_scores

    similarity_results = similarity_results.merge(
        rating_stats,
        on="movieId",
        how="left"
    )

    similarity_results = similarity_results[
        similarity_results["title"] != df_movies.loc[movie_index, "title"]
    ]

    selected_genres = df_movies.loc[movie_index, "genres"]
    selected_genres = set(selected_genres.split("|"))

    def get_shared_genres(recommended_genres):
        recommended_genres = set(recommended_genres.split("|"))
        shared_genres = selected_genres.intersection(recommended_genres)
        return ", ".join(sorted(shared_genres))
    
    similarity_results["shared_genres"] = similarity_results["genres"].apply(get_shared_genres)

    similarity_results = similarity_results.sort_values(
        by=["similarity_score", "rating_count", "average_rating", "title"],
        ascending=[False, False, False, True]
    )

    return similarity_results.head(top_n)

print("Recommendations for Toy Story (1995)")
recommendations = recommend_similar_movies("Toy Story (1995)")
print(recommendations[[
    "title",
    "similarity_score",
    "average_rating",
    "rating_count",
    "shared_genres"
]])
print()

print("Recommendations for Matrix, The (1999)")
recommendations = recommend_similar_movies("Matrix, The (1999)")
print(recommendations[[
    "title",
    "similarity_score",
    "average_rating",
    "rating_count",
    "shared_genres"
]])
print()

print("Recommendations for Jurassic Park (1993)")
recommendations = recommend_similar_movies("Jurassic Park (1993)")
print(recommendations[[
    "title",
    "similarity_score",
    "average_rating",
    "rating_count",
    "shared_genres"
]])
print()