import sqlite3
import pandas as pd

df_movies = pd.read_csv("data/ml-latest-small/movies.csv")
df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

connection = sqlite3.connect(":memory:")

df_movies.to_sql("movies", connection, index=False, if_exists="replace")
df_ratings.to_sql("ratings", connection, index=False, if_exists="replace")

query_most_rated = """
SELECT movies.title, ratings.movieId, COUNT(rating) as total_ratings
FROM ratings
INNER JOIN movies
    ON ratings.movieId = movies.movieId
GROUP BY ratings.movieId, movies.title
ORDER BY total_ratings DESC
LIMIT 10;
"""

query_highest_average = """
SELECT movies.title, AVG(rating) as average_rating, COUNT(rating) as rating_count
FROM ratings
INNER JOIN movies
    ON ratings.movieId = movies.movieId
GROUP BY ratings.movieId, movies.title
HAVING rating_count >= 20
ORDER BY average_rating DESC
LIMIT 10;
"""

most_rated_result = pd.read_sql_query(query_most_rated, connection)
highest_average_result = pd.read_sql_query(query_highest_average, connection)

print(most_rated_result)

print()

print(highest_average_result)

connection.close()