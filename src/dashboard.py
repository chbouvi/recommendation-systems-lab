import pandas as pd
import streamlit as st
import plotly.express as px
from collaborative_filtering import recommend_from_similar_users
from content_based import recommend_similar_movies
from popular_baseline import recommend_popular_movies

outputs_df = pd.read_csv("outputs/evaluation_summary.csv")
df_movies = pd.read_csv("data/ml-latest-small/movies.csv")
df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

def get_valid_seed_movie_titles(df_movies, df_ratings, min_rating=4.0, min_rating_count=10):
    high_ratings = df_ratings[(df_ratings["rating"] >= min_rating)]

    high_rating_count = high_ratings.groupby("movieId")["rating"].agg("count").reset_index()

    high_rating_count.columns = ["movieId", "ratings_count"]

    valid_movie_counts = high_rating_count[
        (high_rating_count["ratings_count"] >= min_rating_count)
    ]

    valid_movies = df_movies[
        (df_movies["movieId"].isin(valid_movie_counts["movieId"]))
    ]

    return valid_movies["title"].to_list()

def get_method_display_name(method):
    method_names = {
        "content": "Content-based filtering",
        "collaborative": "Collaborative filtering",
        "popular": "Popular baseline"
    }

    return method_names.get(method, method)

def get_metric_winners(outputs_df):
    metric_columns = ["precision", "recall", "hit_rate"]
    winners = []

    for k_value in sorted(outputs_df["k"].unique()):
        k_results = outputs_df[outputs_df["k"] == k_value]

        for metric in metric_columns:
            best_row = k_results.loc[k_results[metric].idxmax()]

            winners.append({
                "k": k_value,
                "metric": metric,
                "best_method": get_method_display_name(best_row["method"]),
                "best_value": best_row[metric],
            })

    return pd.DataFrame(winners)

def get_dashboard_interpretation(outputs_df):
    k20_results = outputs_df[outputs_df["k"] == 20]

    best_hit_rate_row = k20_results.loc[k20_results["hit_rate"].idxmax()]
    best_method = get_method_display_name(best_hit_rate_row["method"])
    best_hit_rate = best_hit_rate_row["hit_rate"]

    popular_hit_rate = k20_results[
        k20_results["method"] == "popular"
    ]["hit_rate"].iloc[0]

    content_hit_rate = k20_results[
        k20_results["method"] == "content"
    ]["hit_rate"].iloc[0]

    collaborative_hit_rate = k20_results[
        k20_results["method"] == "collaborative"
    ]["hit_rate"].iloc[0]

    interpretation = [
        f"{best_method} has the best Hit Rate@20 at {best_hit_rate:.4f}."
    ]

    if collaborative_hit_rate > popular_hit_rate:
        interpretation.append(
            "Collaborative filtering beats the popular baseline at Hit Rate@20, which suggests it adds value beyond general popularity."
        )
    else:
        interpretation.append(
            "The popular baseline matches or beats collaborative filtering at Hit Rate@20, which suggests general popularity is difficult to outperform in this setup."
        )

    if content_hit_rate > popular_hit_rate:
        interpretation.append(
            "Content-based filtering beats the popular baseline at Hit Rate@20, which suggests genre similarity helps recover hidden liked movies."
        )
    else:
        interpretation.append(
            "Content-based filtering trails the popular baseline in hidden-movie recovery, but it still supports similarity-based recommendations."
        )

    return interpretation

def get_baseline_comparison(outputs_df):
    popular_df = outputs_df[outputs_df["method"] == "popular"][
        ["k", "precision", "recall", "hit_rate"]
    ]

    popular_df.columns = ["k", "precision_popular", "recall_popular", "hit_rate_popular"]

    non_popular_df = outputs_df[outputs_df["method"] != "popular"][
        ["k", "method", "precision", "recall", "hit_rate"]
    ]

    comparison_df = pd.merge(popular_df, non_popular_df, on="k")

    comparison_df["precision_delta_vs_popular"] = comparison_df["precision"] - comparison_df["precision_popular"]
    comparison_df["recall_delta_vs_popular"] = comparison_df["recall"] - comparison_df["recall_popular"]
    comparison_df["hit_rate_delta_vs_popular"] = comparison_df["hit_rate"] - comparison_df["hit_rate_popular"]

    return comparison_df[["k", "method", "precision_delta_vs_popular", "recall_delta_vs_popular", "hit_rate_delta_vs_popular"]]

st.title("Recommendation Systems Lab")

st.sidebar.header("Choose Metric")

metrics_list = [
    "Precision",
    "Recall",
    "Hit Rate"
]

default = "hit_rate"

select_metric = st.sidebar.selectbox(
    "Metric ",
    metrics_list
)

select_metric = select_metric.strip()

if select_metric == "Precision":
    final_metric = "precision"

elif select_metric == "Recall":
    final_metric = "recall"

elif select_metric == "Hit Rate":
    final_metric = "hit_rate"

else:
    final_metric = default

st.subheader("Evaluation Results")
st.dataframe(outputs_df, hide_index=True)
st.caption("Results are averaged across 100 users and 50 trials per user.")
st.caption("Methods: content = content-based filtering, collaborative = collaborative filtering, popular = popular baseline.")

st.divider()

st.subheader("Best Method by Metric")
st.dataframe(get_metric_winners(outputs_df), hide_index=True)

st.divider()

st.subheader("Performance vs Popular Baseline")
st.dataframe(get_baseline_comparison(outputs_df), hide_index=True)
st.caption("Positive values mean the method outperformed the popular baseline for that metric")

st.divider()

st.subheader("Evaluation Interpretation")

for sentence in get_dashboard_interpretation(outputs_df):
    st.markdown(f"- {sentence}")

st.divider()

highest_metric_by_method = outputs_df.groupby("method")[final_metric].max()
best_method = highest_metric_by_method.idxmax()

best_method = get_method_display_name(best_method)

highest_hit_rate = outputs_df.groupby("k")["hit_rate"].max()
best_hit_rate = highest_hit_rate[20]

col1, col2 = st.columns(2)

with col1:
    st.metric("Users evaluated", outputs_df["num_users"][0])

with col2:
    st.metric("Trials per user", outputs_df["trials_per_user"][0])

col3, col4 = st.columns(2)

with col3:
    st.metric(f"Best method for {select_metric}", best_method)

with col4:
    st.metric("Best Hit Rate@20", best_hit_rate)

st.divider()

outputs_df["k_label"] = outputs_df["k"].astype(str)

fig1 = px.bar(
    outputs_df,
    x="k_label",
    y=final_metric,
    color="method",
    title=f"{select_metric} per K",
    barmode="group"
)

fig1.update_layout(
    xaxis_title="K Value",
    yaxis_title=select_metric,
    xaxis_type="category"
)

st.subheader("Metric Comparison")
st.plotly_chart(fig1, width="stretch")
st.caption(f"{best_method} performs best for {select_metric}@K in this evaluation. Compare it with the popular baseline to see whether other methods add value beyond popularity.")

st.divider()

st.subheader("Example Recommendations")

seed_title = st.selectbox(
    "Choose seed movie...", 
    get_valid_seed_movie_titles(df_movies, df_ratings), 
)

st.caption(f"Seed movie: {seed_title}")

content_recommendations = recommend_similar_movies(seed_title)
content_recommendations = content_recommendations[["title"]]
content_recommendations.columns = ["Movie Title"]

top_movies_collaborative, _ = recommend_from_similar_users(seed_title)
top_movies_collaborative = top_movies_collaborative[["title"]]
top_movies_collaborative.columns = ["Movie Title"]

top_popular_movies = recommend_popular_movies(seed_title)
top_popular_movies = top_popular_movies[["title"]]
top_popular_movies.columns = ["Movie Title"]

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Content-based")
    st.dataframe(content_recommendations, hide_index=True)
with col2:
    st.caption("Collaborative filtering")
    if top_movies_collaborative.empty:
        st.info("Not enough similar-user data for this movie. Try a more popular seed movie.")
    else:
        st.dataframe(top_movies_collaborative, hide_index=True)
with col3:
    st.caption("Popular baseline")
    st.dataframe(top_popular_movies, hide_index=True)
