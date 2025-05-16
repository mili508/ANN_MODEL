import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Title
st.title("🏦 Loan Approval Prediction Using ANN")

# Load model and scaler
@st.cache_resource
def load_resources():
    model = load_model("loan_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_resources()

# Input fields
st.header("🔢 Enter Applicant Details")

no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Encode categorical variables
education_binary = 1 if education == "Graduate" else 0
self_employed_binary = 1 if self_employed == "Yes" else 0

# Create DataFrame
data = {
    'no_of_dependents': no_of_dependents,
    'education': education_binary,
    'self_employed': self_employed_binary,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

input_df = pd.DataFrame([data])

# Define column order if needed
columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
           'loan_amount', 'loan_term', 'cibil_score',
           'residential_assets_value', 'commercial_assets_value',
           'luxury_assets_value', 'bank_asset_value']
input_df = input_df[columns]

# Predict
if st.button("🚀 Predict Loan Approval"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0][0]
        output = "✅ Loan Approved" if prediction > 0.5 else "❌ Loan Rejected"
        st.success(f"Prediction: {output}")
        st.write(f"Confidence Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")