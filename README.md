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

## How It Works

The project currently includes two recommendation approaches. 

The content-based recommender looks at each movie's genres and compares movies based on how many genres they have in common. For example, if two movies both include `Adventure`, `Animation`, and `Comedy`, they get a higher similarity score than movies with fewer overlapping genres. If multiple movies have the same similarity score, the recommender uses rating count and average rating to help decide the order.

The collaborative filtering recommender uses user behavior instead of movie genres. It finds users who rated a selected movie highly, then recommends other movies those users also rated highly. It also tracks how many similar users liked each movie and their average rating for that movie.

For `Toy Story (1995)`, the content-based recommender returns movies like `Shrek`, `Toy Story 2`, and `Monsters, Inc.`, while the collaborative filtering recommender returns movies like `The Shawshank Redemption`, `Forrest Gump`, and `Star Wars: Episode IV - A New Hope`.

This illustrates the difference between recommending based on features and recommending based on user behavior.

## Testing

The project uses pytest for automated tests. Tests live in the root-level `tests/` folder, separate from the recommender code in `src/`.

Run the test suite with:

```bash
pytest tests/
```

Current tests check that both recommenders handle valid and invalid movie titles, return expected columns, exclude the original movie, and respect parameters like `top_n` and `min_rating`.

## MovieLens

MovieLens is a public movie ratings dataset commonly used for recommendation systems. It includes user ratings, movie titles, and movie genres. This makes it a good dataset for comparing recommendation approaches.

## Planned Features

- Improve collaborative filtering recommendation ranking
- Expand comparison between content-based filtering and collaborative filtering
- Add recommendation quality metrics
- Build a simple Streamlit dashboard
- Add tests for evaluation metrics and dashboard logic

## Planned Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- Plotly
- Pytest
