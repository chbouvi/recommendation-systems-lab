import pandas as pd

df_movies = pd.read_csv("data/ml-latest-small/movies.csv")
df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

def recommend_popular_movies(seed_title, min_rating=4.0, top_n=10, ratings_df=None, movies_df=None):
    if ratings_df is None:
        ratings_df = df_ratings
    if movies_df is None:
        movies_df = df_movies
    
    movie_row = movies_df[movies_df["title"] == seed_title]

    if movie_row.empty:
        return None

    movie_id = movie_row["movieId"].iloc[0]

    top_rated = ratings_df[
        (ratings_df["rating"] >= min_rating) &
        (ratings_df["movieId"] != movie_id)
    ]

    top_movies = (
        top_rated.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .reset_index()
    )

    top_movies.columns = ["movieId", "high_rating_count", "average_rating"]

    top_movies = top_movies.merge(
        movies_df[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    top_movies = top_movies.sort_values(
        by=["high_rating_count", "average_rating", "title"],
        ascending=[False, False, True]
    )

    top_movies = top_movies.head(top_n)

    return top_movies

if __name__ == "__main__":
    seed_title = "Toy Story (1995)"

    top_movies = recommend_popular_movies(seed_title)

    if top_movies is None:
        print("Movie not found.")

    else:
        print(f"Popular baseline recommendations for {seed_title}")
        print(top_movies[["title", "high_rating_count", "average_rating"]])

