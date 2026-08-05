import pandas as pd
import streamlit as st
import plotly.express as px

outputs_df = pd.read_csv("outputs/evaluation_summary.csv")

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

highest_metric_by_method = outputs_df.groupby("method")[final_metric].max()
best_method = highest_metric_by_method.idxmax()

if best_method == "collaborative":
    best_method = "Collaborative filtering"
elif best_method == "content":
    best_method = "Content-based"

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
st.caption(f"{best_method} has higher {select_metric}@K in this run.")