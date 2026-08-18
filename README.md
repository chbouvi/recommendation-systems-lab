# Recommendation Systems Lab

A machine learning project that uses the MovieLens dataset to compare different recommendation approaches.

## MVP

A recommendation lab that explains why items are recommended, compares content-based filtering and collaborative filtering, and evaluates recommendation quality.

## Current Progress

- Explored MovieLens ratings and movie metadata
- Built a genre-based content recommender
- Added shared-genre explanations for recommendations
- Added rating count and average rating as tie-breakers for ranked recommendations
- Built an initial collaborative filtering baseline using highly rated movies from similar users
- Added average similar-user rating to collaborative recommendations
- Added pytest tests for content-based and collaborative filtering behavior
- Added repeated one-user evaluation across multiple K values using hidden liked movies and averaged Precision@K, Recall@K, and Hit Rate@K
- Added multi-user evaluation across multiple K values
- Added evaluation support for recommender method selection across content-based, collaborative, and popular baseline methods
- Added normalized collaborative scoring using the ratio of similar users who liked each movie and the average rating
- Added evaluation summaries saved as CSV
- Added a Streamlit dashboard for viewing evaluation outputs, summary metrics, and metric comparisons
- Added interactive dashboard examples for comparing content-based and collaborative recommendations from a selected seed movie
- Filtered dashboard seed movie choices to movies with enough high ratings for collaborative filtering
- Added a popular baseline recommender based on movies rated highest globally
- Compared content-based filtering, collaborative filtering, and the popular baseline in evaluation

## How It Works

The project currently includes three recommendation approaches.

The content-based recommender looks at each movie's genres and compares movies based on how many genres they have in common. For example, if two movies both include `Adventure`, `Animation`, and `Comedy`, they get a higher similarity score than movies with fewer overlapping genres. If multiple movies have the same similarity score, the recommender uses rating count and average rating to help decide the order.

The collaborative filtering recommender uses user behavior instead of movie genres. It finds users who rated a selected movie highly, then recommends other movies those users also rated highly. It also tracks how many similar users liked each movie, their average rating for that movie, and a normalized collaborative score using the ratio of similar users who liked each movie and the average rating.

For `Toy Story (1995)`, the content-based recommender returns movies like `Shrek`, `Toy Story 2`, and `Monsters, Inc.`, while the collaborative filtering recommender returns movies like `The Shawshank Redemption`, `Forrest Gump`, and `Star Wars: Episode IV - A New Hope`.

This illustrates the difference between recommending based on features and recommending based on user behavior.

The popular baseline recommends movies with the most high ratings globally, excluding the selected seed movie. It doesn't use genres or similar-user behavior like the other recommenders, so it gives a simple comparison to use when judging whether the other methods are better in evaluation than general popularity.

## Evaluation

The project includes an evaluation setup using a hidden movie. It randomly hides one movie a user rated highly (rating >= 4.0), generates recommendations from another liked movie, and checks whether the hidden movie appears in the top K recommendations.

The evaluation computes average Precision@K, Recall@K, and Hit Rate@K across multiple trials and K values.

The evaluation also averages metrics across multiple users, which gives a broader view than a single-user demo.

The evaluation can run the content-based, collaborative, or popular baseline recommender across random user samples, multiple trials per user, and multiple K values.

The evaluation script also saves a summary CSV with each recommender method, K value, precision, recall, hit rate, number of users, and trials per user.

The latest evaluation compares content-based filtering, collaborative filtering, and the popular baseline. Collaborative filtering performed best on hidden-movie recovery, the popular baseline placed second, and content-based filtering placed third. This suggests that collaborative filtering is learning useful user-behavior patterns beyond simple movie popularity, while content-based filtering remains helpful for recommendations similar to the seed movie.

## SQL Exploration

I added a small SQLite exploration script to practice SQL directly on the MovieLens dataset used in this project. The script loads the `movies.csv` and `ratings.csv` files into in-memory SQLite tables and runs queries to answer some basic questions.

Current questions done:
- What are the top 10 most-rated movies?
- What are the top 10 highest-average-rated movies with at least 20 ratings?
- What genres appear most often among highly-rated movies?
- What are the top 10 users by number of ratings?

This helped connect SQL concepts like `JOIN`, `GROUP BY`, `COUNT`, `AVG`, `HAVING`, and `ORDER BY` to the dataset. Originally, a limitation I had was counting full genre strings rather than individual genres. To fix this, I added a new table by splitting the genre strings into individual genre rows. Now, the genre query counts individual genres among highly rated movies.

## Testing

The project uses pytest for automated tests. Tests live in the `tests/` folder, separate from the recommender code in `src/`.

Run the test suite with:

```bash
pytest tests/
```

Current tests check that recommenders handle valid and invalid inputs, return expected outputs, respect parameters like `top_n` and `min_rating`, compute evaluation metrics correctly, hold out hidden movies during evaluation, run evaluation across multiple K values and users, and support method selection between content-based, collaborative, and popular baseline recommendations.

## Dashboard

The dashboard displays evaluation results, summary metrics, metric comparison charts, and example recommendations for a selected seed movie.

Run the Streamlit dashboard with:

```bash
streamlit run src/dashboard.py
```

## MovieLens

MovieLens is a public movie ratings dataset commonly used for recommendation systems. It includes user ratings, movie titles, and movie genres.

## Planned Features

- Explain method comparison results more clearly
- Continue improving collaborative filtering recommendation ranking
- Improve the Streamlit dashboard with more charts and examples
- Expand tests for evaluation metrics and dashboard logic

## Planned Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- Plotly
- Pytest
