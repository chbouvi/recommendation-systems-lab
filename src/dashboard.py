import pandas as pd
import streamlit as st

outputs_df = pd.read_csv("outputs/evaluation_summary.csv")

st.title("Recommendation Systems Lab")

st.dataframe(outputs_df, hide_index=True)