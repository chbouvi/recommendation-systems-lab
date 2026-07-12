import pandas as pd

df_movies = pd.read_csv("data/ml-latest-small/movies.csv")

def recommend_similar_movies(movie_title, top_n=10):
    movie_row = df_movies[df_movies["title"] == movie_title]

    if movie_row.empty:
        print("Movie not found.")
        return
    
    movie_index = movie_row.index[0]
    
    genre_matrix = df_movies["genres"].str.get_dummies(sep="|")

    selected_movie = genre_matrix.loc[movie_index]

    similarity_scores = genre_matrix.dot(selected_movie)

    similarity_results = df_movies[["title", "genres"]].copy()
    similarity_results["similarity_score"] = similarity_scores

    similarity_results = similarity_results[
        similarity_results["title"] != df_movies.loc[movie_index, "title"]
    ]

    similarity_results = similarity_results.sort_values(
        by=["similarity_score", "title"],
        ascending=[False, True]
    )

    return similarity_results.head(top_n)

print("Recommendations for Toy Story (1995)")
print(recommend_similar_movies("Toy Story (1995)"))
print()

print("Recommendations for Matrix, The (1999)")
print(recommend_similar_movies("Matrix, The (1999)"))
print()

print("Recommendations for Jurassic Park (1993)")
print(recommend_similar_movies("Jurassic Park (1993)"))
print()