import pandas as pd
import streamlit as st
import plotly.express as px
from collaborative_filtering import recommend_from_similar_users
from content_based import recommend_similar_movies
from popular_baseline import recommend_popular_movies

st.set_page_config(
    page_title="Recommendation Systems Lab",
    layout="wide",
)

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

    comparison_df["method"] = comparison_df["method"].map(get_method_display_name)

    return comparison_df[["k", "method", "precision_delta_vs_popular", "recall_delta_vs_popular", "hit_rate_delta_vs_popular"]]

st.title("Recommendation Systems Lab")
st.write("Comparing content-based, collaborative, and popularity recommendations on MovieLens.")

k_values = sorted(outputs_df["k"].unique())
metric_columns = {"Hit Rate": "hit_rate", "Precision": "precision", "Recall": "recall"}
num_users = int(outputs_df["num_users"].iloc[0])
trials_per_user = int(outputs_df["trials_per_user"].iloc[0])

st.subheader("Evaluation")
st.caption(
    "Each trial hides a rating for a movie the user liked, then uses another liked movie as the seed."
)
control_cols = st.columns(2)
with control_cols[0]:
    selected_metric = st.selectbox("Metric", list(metric_columns))
with control_cols[1]:
    focus_k = st.selectbox("Top K", k_values, index=len(k_values) - 1)

metric = metric_columns[selected_metric]
chart_df = outputs_df.assign(method_name=outputs_df["method"].map(get_method_display_name))
focus_results = chart_df[chart_df["k"] == focus_k].copy()
focus_results = focus_results.sort_values(metric, ascending=False)
method_colors = {
    "Content-based filtering": "#2e5e9e",
    "Collaborative filtering": "#127c68",
    "Popular baseline": "#ae6135",
}

chart_col, results_col = st.columns([2, 1], gap="large")
with chart_col:
    fig = px.line(
        chart_df,
        x="k",
        y=metric,
        color="method_name",
        markers=True,
        color_discrete_map=method_colors,
        labels={"k": "K", metric: selected_metric, "method_name": "Method"},
    )
    fig.update_traces(hovertemplate="K=%{x}<br>Score=%{y:.1%}<extra>%{fullData.name}</extra>")
    fig.update_layout(
        height=370,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(tickmode="array", tickvals=k_values),
        yaxis=dict(tickformat=".1%", rangemode="tozero"),
        legend=dict(orientation="h", title_text="", y=1.16),
    )
    st.plotly_chart(fig, width="stretch")

with results_col:
    st.markdown(f"**{selected_metric}@{focus_k}**")
    focus_table = focus_results[["method_name", metric]].rename(
        columns={"method_name": "Method", metric: selected_metric}
    )
    focus_table[selected_metric] = focus_table[selected_metric].map(lambda value: f"{value:.1%}")
    st.dataframe(focus_table, hide_index=True, width="stretch")
    st.caption(f"{num_users} users, {trials_per_user} trials per user.")

st.divider()
st.subheader("Example recommendations")
seed_title = st.selectbox(
    "Seed movie",
    get_valid_seed_movie_titles(df_movies, df_ratings),
)

content_recommendations = recommend_similar_movies(seed_title)
content_recommendations = content_recommendations[["title", "shared_genres"]]
content_recommendations.columns = ["Movie Title", "Shared Genres"]

top_movies_collaborative, _ = recommend_from_similar_users(seed_title)
top_movies_collaborative = top_movies_collaborative[
    ["title", "similar_user_likes", "average_similar_user_rating"]
]
top_movies_collaborative.columns = [
    "Movie Title", "Similar User Likes", "Average Similar User Rating"
]

top_popular_movies = recommend_popular_movies(seed_title)
top_popular_movies = top_popular_movies[["title"]]
top_popular_movies.columns = ["Movie Title"]

content_tab, collaborative_tab, popular_tab = st.tabs(
    ["Content-based", "Collaborative", "Popular baseline"]
)
with content_tab:
    st.dataframe(content_recommendations, hide_index=True, width="stretch", height=360)
with collaborative_tab:
    if top_movies_collaborative.empty:
        st.info("Not enough similar-user data for this movie. Try another seed.")
    else:
        st.dataframe(top_movies_collaborative, hide_index=True, width="stretch", height=360)
with popular_tab:
    st.dataframe(top_popular_movies, hide_index=True, width="stretch", height=360)

st.divider()
with st.expander("Full evaluation tables"):
    st.caption(
        "With one hidden target per trial, Recall@K equals Hit Rate@K and "
        "Precision@K equals Hit Rate@K / K."
    )
    results_tab, baseline_tab, winners_tab = st.tabs(
        ["Summary", "Compared with popularity", "Best method by metric"]
    )
    with results_tab:
        st.dataframe(outputs_df, hide_index=True, width="stretch")
    with baseline_tab:
        st.caption("Positive values mean a method scored above the popularity baseline.")
        st.dataframe(get_baseline_comparison(outputs_df), hide_index=True, width="stretch")
    with winners_tab:
        st.dataframe(get_metric_winners(outputs_df), hide_index=True, width="stretch")
