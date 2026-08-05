## Recommendation Systems Lab Explanation Notes

Recommendation Systems Lab is a MovieLens-based machine learning project where I compare content-based filtering and collaborative filtering.

The content-based recommender uses movie genres. For each movie, it looks at the genres listed in the dataset, like `Comedy`, `Drama`, `Adventure`, or `Sci-Fi`. It recommends movies that share the most genres with the selected movie. For example, if two movies are both listed as `Adventure` and `Sci-Fi`, they are treated as more similar than movies with no genre overlap.

The collaborative filtering recommender uses user rating behavior instead of movie genres. It finds users who rated a selected movie highly, then looks at what other movies those users also rated highly. Those movies become recommendations. I improved the ranking by using a normalized score based on the share of similar users who liked each movie and the average rating from those users.

To evaluate the recommenders, I use a hidden-movie setup. For each trial, I hide one movie a user rated highly, generate recommendations from another movie that user liked, and check whether the hidden movie appears in the top `K` recommendations.

I compare the recommenders using `Precision@K`, `Recall@K`, and `Hit Rate@K` across multiple users, multiple trials, and multiple `K` values. The evaluation script can run both recommendation methods and saves a summary CSV of the results.

I also built an initial Streamlit dashboard to make the evaluation results easier to view. The dashboard loads the saved evaluation summary CSV, displays the results table, and shows grouped bar charts comparing content-based and collaborative filtering across different K values and metrics. It also displays summary metrics to highlight the number of users evaluated, trials per user, the best method for the selected metric, and the best Hit Rate@20.