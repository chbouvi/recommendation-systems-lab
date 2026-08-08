# Evaluation Notes

In this project, a movie is considered **relevant** if the user rated it highly. For example, I can treat ratings greater than or equal to `4.0` as relevant.

The goal of evaluation is to compare the recommender's top `K` recommendations against the movies the user actually liked.

## Precision@K

Precision@K measures how many of the top `K` recommendations were actually relevant.

```text
precision@K = relevant recommendations in the top K / K
```

Example:

Recommended top 5 movies: 5

Relevant recommendations in top 5: 2

```text
precision@5 = 2 / 5 = 0.40
```

In terms of this project:

```text
Of the movies I recommended, how many did the user actually like?
```

### True Positives

Movies that were recommended in the top `K` and were actually relevant.

### False Positives

Movies that were recommended in the top `K` but were not actually relevant.

## Recall@K

Recall@K measures how many of the user's relevant movies were recovered in the top `K` recommendations.

```text
recall@K = relevant recommendations in the top K / total relevant movies for that user
```

Example:

Relevant movies for the user: 3

Relevant movies found in top 5: 2

```text
recall@5 = 2 / 3 = 0.67
```

In terms of this project:

```text
Of the movies the user actually liked, how many did the recommender find?
```

### False Negatives

Movies that were relevant to the user but did not appear in the top `K` recommendations.

## Hit Rate@K

Hit Rate@K measures whether the recommender found at least one relevant movie in the top `K`.

```text
hit rate@K = 1 if at least one relevant item appears in the top K, otherwise 0
```

Example:

If at least one top 5 recommendation is relevant:

```text
hit rate@5 = 1
```

If none of the top 5 recommendations are relevant:

```text
hit rate@5 = 0
```

In terms of this project:

```text
Did the recommender find at least one movie the user actually liked?
```

## Simple Example

Top 5 recommendations:
[A, B, C, D, E]

Movies the user actually liked:
[B, D, X]

Relevant recommendations:
[B, D]

```text
precision@5 = 2 / 5 = 0.40
recall@5 = 2 / 3 = 0.67
hit rate@5 = 1
```

## Mental Model

- Precision@K: How many of my recommendations were good?
- Recall@K: How many of the user's liked movies did I recover?
- Hit Rate@K: Did I get at least one good recommendation?

## Comparing Recommenders

The evaluation now uses the same setup to compare content-based filtering and collaborative filtering.

Content-based:
Uses one seed movie title and recommends movies with similar genres.

Collaborative:
Uses one seed movie title, finds users who liked that movie, then recommends other movies those users liked.

## Initial Comparison Result

Using users 1-5, 50 trials per user, and K values of 5, 10, and 20, collaborative filtering was able to find hidden liked movies more often than the content-based recommender.

## Larger Comparison Result

Using 50 randomly selected users, 50 trials per user, and K values of 5, 10, and 20, collaborative filtering found hidden liked movies more often than the content-based recommender again. 

This evaluation ran 2,500 hidden-movie trials per recommender.

Example output: 

```text
Content-based Hit Rate@20: 0.0564
Collaborative Hit Rate@20: 0.2212
```

This makes sense because the collaborative recommender uses patterns in user ratings, while the content-based recommender only uses movie genres.

The evaluation output includes a summary table for comparing the recommender results.

## 100-User Comparison Result

Using 100 randomly selected users, 50 trials per user, and K values of 5, 10, and 20, collaborative filtering still found hidden liked movies more often than content-based.

The evaluation ran 5,000 hidden-movie trials per recommender.

Example output:

```text
Content-based Hit Rate@20: 0.0488
Collaborative Hit Rate@20: 0.1762
```

## Collaborative Ranking Explanation

The collaborative recommender first counts how many similar users liked each movie.

To make the count easier to compare, I convert it into a ratio:

```text
similar_user_like_ratio = similar_user_likes / total_similar_users
```

Then I combine that ratio with the average rating from similar users:

```text
collaborative_score = similar_user_like_ratio * average_similar_user_rating
```

## Comparison Result with New Collaborative Ranking

After normalizing the collaborative score, collaborative filtering still performed better than content-based filtering in the evaluation. The normalized score is also easier to interpret because it uses the share of similar users who liked each movie instead of only the count.

## Next Direction

- Explain the method comparison results more clearly.
- Review the collaborative filtering scoring formula.
- Improve collaborative filtering beyond the dashboard by adding stronger ranking logic.
- Improve the Streamlit dashboard with more charts and examples.
- Expand tests for evaluation metrics and dashboard logic.
