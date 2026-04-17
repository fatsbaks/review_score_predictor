import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("random_forest_model.pkl")

st.title("Review Score Prediction App")
st.write("Enter product details to predict the review score:")

price = st.number_input("Product Price", min_value = 0.0, step = 0.01)
freight_value = st.number_input("Freight Price", min_value = 0.0, step = 0.01)
delivery_days = st.number_input("Delivery Days", min_value = 0, step = 1)

features = np.array([[price, freight_value, delivery_days]])

predicted_score = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

scores = model.classes_  # possible review scores (e.g., [1,2,3,4,5])
prob_df = pd.DataFrame({
    "Review Score": scores,
    "Probability (%)": (probabilities * 100).round(2)
})

st.subheader("Prediction Result:")
st.write(f"Predicted review score: **{predicted_score}**")

st.table(prob_df)