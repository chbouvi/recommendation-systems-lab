import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
df_movies = pd.read_csv("data/ml-latest-small/movies.csv")

def find_users_who_like_movie(movie_title, min_rating):
    movie_row = df_movies[df_movies["title"] == movie_title]

    if movie_row.empty:
        return None, None
    
    movie_id = movie_row["movieId"].iloc[0]

    liked_ratings = df_ratings[
        (df_ratings["movieId"] == movie_id) &
        (df_ratings["rating"] >= min_rating)
    ]

    return liked_ratings["userId"].unique(), movie_id

def recommend_from_similar_users(movie_title, min_rating=4.0, top_n=10):
    users, movie_id = find_users_who_like_movie(movie_title, min_rating)

    if users is None:
        return None, None

    similar_user_ratings = df_ratings[
        (df_ratings["userId"].isin(users)) &
        (df_ratings["rating"] >= min_rating) & 
        (df_ratings["movieId"] != movie_id)
    ]

    movie_stats = (
        similar_user_ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .reset_index()
    )

    movie_stats.columns = ["movieId", "similar_user_likes", "average_similar_user_rating"]

    movie_stats = movie_stats.sort_values(
        by=["similar_user_likes", "average_similar_user_rating"],
        ascending=[False, False]
    )

    top_movie_stats = movie_stats.head(top_n)

    top_movies = top_movie_stats.merge(
        df_movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return top_movies, len(users)

top_movies, user_amount = recommend_from_similar_users("Toy Story (1995)")

if top_movies is None:
    print("Movie not found.")
else:
    print("Collaborative recommendations for Toy Story (1995)")
    print(f"Number of users who liked Toy Story: {user_amount}")
    print()
    print(top_movies[["title", "similar_user_likes", "average_similar_user_rating", "genres"]])



