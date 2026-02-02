import streamlit as st
import pandas as pd
from wellnessz_runtime import predict_clients, generate_explanation

st.title("WellnessZ — Physiology Based Coaching")

file = st.file_uploader("Upload client CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    results = predict_clients(df)

    st.subheader("Client Results")
    st.dataframe(results)

    cid = st.selectbox("Select client", results["client_id"])

    if st.button("Generate Explanation"):
        row = results[results["client_id"] == cid].iloc[0]
        st.write(generate_explanation(row))
