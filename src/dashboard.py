import pandas as pd
import streamlit as st
import plotly.express as px

outputs_df = pd.read_csv("outputs/evaluation_summary.csv")

st.title("Recommendation Systems Lab")

st.dataframe(outputs_df, hide_index=True)

outputs_df["k_label"] = outputs_df["k"].astype(str)

fig1 = px.bar(
    outputs_df,
    x="k_label",
    y="hit_rate",
    color="method",
    title="Hit Rate per K",
    barmode="group"
)

fig1.update_layout(
    xaxis_title="K Value",
    yaxis_title="Hit Rate",
    xaxis_type="category"
)

st.plotly_chart(fig1, width="stretch")