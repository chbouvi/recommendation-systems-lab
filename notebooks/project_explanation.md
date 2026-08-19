## Recommendation Systems Lab Explanation Notes

Recommendation Systems Lab is a MovieLens-based machine learning project where I compare content-based filtering, collaborative filtering, and a popularity baseline.

The content-based recommender uses movie genres. For each movie, it looks at the genres listed in the dataset, like `Comedy`, `Drama`, `Adventure`, or `Sci-Fi`. It recommends movies that share the most genres with the selected movie. For example, if two movies are both listed as `Adventure` and `Sci-Fi`, they are treated as more similar than movies with no genre overlap.

The collaborative filtering recommender uses user rating behavior instead of movie genres. It finds users who rated a selected movie highly, then looks at what other movies those users also rated highly. Those movies become recommendations. I improved the ranking by using a normalized score based on the share of similar users who liked each movie and the average rating from those users.

The popular baseline recommends movies with the most high ratings globally, excluding the selected seed movie. It doesn't use genres or similar-user behavior, so it gives a simple benchmark for judging whether the other recommendation methods are actually learning.

To evaluate the recommenders, I use a hidden-movie setup. For each trial, I hide one movie a user rated highly, generate recommendations from another movie that user liked, and check whether the hidden movie appears in the top `K` recommendations.

I compare the recommenders using `Precision@K`, `Recall@K`, and `Hit Rate@K` across multiple users, multiple trials, and multiple `K` values. The evaluation script can run all three recommendation methods and saves a summary CSV of the results.

I also built an initial Streamlit dashboard to make the evaluation results easier to view. The dashboard loads the saved evaluation summary CSV, displays the results table, and shows grouped bar charts comparing content-based filtering, collaborative filtering, and the popular baseline across different K values and metrics. It also displays summary metrics to highlight the number of users evaluated, trials per user, the best method for the selected metric, and the best Hit Rate@20. In addition, there is an interactive example section where users can choose a seed movie and compare content-based and collaborative recommendations side by side. The example section filters seed movies to those with enough high ratings, and the dashboard shows an info message when collaborative filtering does not have enough similar-user data for a selected movie.

In the latest evaluation, collaborative filtering performs best on hidden-movie recovery, the popular baseline places second, and content-based filtering places third. This suggests collaborative filtering is capturing useful user-behavior patterns beyond general popularity, while content-based filtering may still feel more directly similar to the seed movie.

I also added SQLite exploration for the MovieLens data, including most-rated movies, highest-average-rated movies with at least 20 ratings, highly rated genres, and most active users.