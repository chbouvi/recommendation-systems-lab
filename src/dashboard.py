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