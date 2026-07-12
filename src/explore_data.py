import pandas as pd

df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
df_movies = pd.read_csv("data/ml-latest-small/movies.csv")

num_ratings = len(df_ratings)
num_movies = len(df_movies)
num_users = df_ratings["userId"].nunique()
top10_most_rated = (
    df_ratings["movieId"]
    .value_counts()
    .head(10)
    .reset_index()
)

top10_most_rated.columns = ["movieId", "rating_count"]

top10_most_rated = top10_most_rated.merge(
    df_movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

rating_distribution = df_ratings["rating"].value_counts().sort_index()

print(f"Number of ratings: {num_ratings}\n")
print(f"Number of movies: {num_movies}\n")
print(f"Number of users: {num_users}\n")

print(f"Missing values in ratings:")
print(df_ratings.isnull().sum())
print()

print(f"Missing values in movies:")
print(df_movies.isnull().sum())
print()

print(f"Top 10 most-rated movies:")
print(top10_most_rated[["title", "rating_count"]])

print("Rating distribution:")
print(rating_distribution)
print()