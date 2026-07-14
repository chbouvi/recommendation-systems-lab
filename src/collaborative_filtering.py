import pandas as pd

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

print(users)
print(f"Number of users who liked Toy Story: {len(users)}")

print(similar_user_ratings.head())
print(len(similar_user_ratings))

top_movie_ids = similar_user_ratings["movieId"].value_counts().head(10)
print(top_movie_ids)


