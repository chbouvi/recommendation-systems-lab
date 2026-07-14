import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
df_movies = pd.read_csv("data/ml-latest-small/movies.csv")

def find_users_who_like_movie(movie_title, min_rating):
    movie_row = df_movies[df_movies["title"] == movie_title]

    if movie_row.empty:
        print("Movie not found.")
        return
    
    movie_id = movie_row["movieId"].iloc[0]

    liked_ratings = df_ratings[
        (df_ratings["movieId"] == movie_id) &
        (df_ratings["rating"] >= min_rating)
    ]

    return liked_ratings["userId"].unique(), movie_id

min_rating = 4.0
users, movie_id = find_users_who_like_movie("Toy Story (1995)", min_rating)

similar_user_ratings = df_ratings[
    (df_ratings["userId"].isin(users)) &
    (df_ratings["rating"] >= min_rating) & 
    (df_ratings["movieId"] != movie_id)
]

top_movie_ids = similar_user_ratings["movieId"].value_counts().head(10)
top_movies = top_movie_ids.reset_index()
top_movies.columns = ["movieId", "similar_user_likes"]
top_movies = top_movies.merge(
    df_movies[["movieId", "title", "genres"]],
    on="movieId",
    how="left"
)

print("Collaborative recommendations for Toy Story (1995)")
print(f"Number of users who liked Toy Story: {len(users)}")
print()
print(top_movies[["title", "similar_user_likes", "genres"]])


