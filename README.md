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

## How It Works

The current recommender is content-based. It looks at each movie's genres and compares movies based on how many genres they have in common.

For example, if two movies both include `Adventure`, `Animation`, and `Comedy`, they get a higher similarity score than movies with fewer overlapping genres.

If multiple movies have the same similarity score, the recommender uses rating count and average rating to decide the order. Genre similarity is still the main signal, but ratings help break ties.

The output also shows the shared genres so the recommendations are easier to understand.

## MovieLens

MovieLens is a public movie ratings dataset commonly used for recommendation systems. It includes user ratings, movie titles, and movie genres. This makes it a good dataset for comparing recommendation approaches.

## Planned Features

- Improve collaborative filtering recommendation ranking
- Compare recommendation methods (content-based filtering vs. collaborative filtering)
- Evaluate recommendation quality with metrics
- Build a simple Streamlit dashboard
- Add tests for recommendation logic

## Planned Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- Plotly
- Pytest
