import pandas as pd

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

if __name__ == "__main__":
    df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    print(build_like_matrix(df_ratings))
