from item_to_item import build_like_matrix
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


if __name__ == "__main__":
    matrix = build_like_matrix(df_ratings)
    result = get_scaled_popularity_scores(matrix)
    print(result)
