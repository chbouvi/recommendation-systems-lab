import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
df_movies = pd.read_csv("data/ml-latest-small/movies.csv")

def find_users_who_like_movie(movie_title, min_rating, ratings_df=None, movies_df=None):
    if ratings_df is None:
        ratings_df = df_ratings
    if movies_df is None:
        movies_df = df_movies

    movie_row = movies_df[movies_df["title"] == movie_title]

    if movie_row.empty:
        return None, None
    
    movie_id = movie_row["movieId"].iloc[0]

    liked_ratings = ratings_df[
        (ratings_df["movieId"] == movie_id) &
        (ratings_df["rating"] >= min_rating)
    ]

    return liked_ratings["userId"].unique(), movie_id

def recommend_from_similar_users(movie_title, min_rating=4.0, top_n=10, min_similar_user_likes=2, ratings_df=None, movies_df=None):
    if ratings_df is None:
        ratings_df = df_ratings
    if movies_df is None:
        movies_df = df_movies

    users, movie_id = find_users_who_like_movie(movie_title, min_rating, ratings_df, movies_df)

    if users is None:
        return None, None

    similar_user_ratings = ratings_df[
        (ratings_df["userId"].isin(users)) &
        (ratings_df["rating"] >= min_rating) & 
        (ratings_df["movieId"] != movie_id)
    ]

    movie_stats = (
        similar_user_ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .reset_index()
    )

    movie_stats.columns = ["movieId", "similar_user_likes", "average_similar_user_rating"]

    movie_stats = movie_stats[movie_stats["similar_user_likes"] >= min_similar_user_likes]

    movie_stats["similar_user_like_ratio"] = movie_stats["similar_user_likes"] / len(users)

    movie_stats["collaborative_score"] = movie_stats["similar_user_like_ratio"] * movie_stats["average_similar_user_rating"]

    movie_stats = movie_stats.sort_values(
        by=["collaborative_score", "similar_user_likes", "average_similar_user_rating"],
        ascending=[False, False, False]
    )

    top_movie_stats = movie_stats.head(top_n)

    top_movies = top_movie_stats.merge(
        movies_df[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return top_movies, len(users)

if __name__ == "__main__":
    top_movies, user_amount = recommend_from_similar_users("Toy Story (1995)")

    if top_movies is None:
        print("Movie not found.")
    else:
        print("Collaborative recommendations for Toy Story (1995)")
        print(f"Number of users who liked Toy Story: {user_amount}")
        print()
        print(top_movies[["title", "collaborative_score", "similar_user_like_ratio", "similar_user_likes", "average_similar_user_rating", "genres"]])



