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