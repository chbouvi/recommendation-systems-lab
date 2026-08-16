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

query_highly_rated_genres = """
SELECT genres, COUNT(genres) as genre_count
FROM movies
INNER JOIN ratings
    ON movies.movieId = ratings.movieId
WHERE ratings.rating >= 4.0
GROUP BY genres
ORDER BY genre_count DESC
LIMIT 10;
"""

query_most_active_users = """
SELECT userId, COUNT(rating) as rating_count
FROM ratings
GROUP BY userId
ORDER BY rating_count DESC
LIMIT 10;
"""

most_rated_result = pd.read_sql_query(query_most_rated, connection)
highest_average_result = pd.read_sql_query(query_highest_average, connection)
highly_rated_genres_result = pd.read_sql_query(query_highly_rated_genres, connection)
most_active_users_result = pd.read_sql_query(query_most_active_users, connection)

print("Top 10 most-rated movies")
print(most_rated_result)

print("-" * 60)

print("Top 10 highest-average-rated movies with at least 20 ratings")
print(highest_average_result)

print("-" * 60)

print("Most common highly rated genre strings")
print(highly_rated_genres_result)

print("-" * 60)

print("Top 10 most active users")
print(most_active_users_result)

connection.close()