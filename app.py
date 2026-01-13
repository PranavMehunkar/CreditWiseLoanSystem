# app.py

import streamlit as st
import pandas as pd
import pickle

# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_order = pickle.load(open("feature_order.pkl", "rb"))
    return model, encoder, scaler, feature_order

model, encoder, scaler, feature_order = load_artifacts()

st.title("🏦 CreditWise Loan Approval System")

# =========================
# User Inputs
# =========================
Age = st.number_input("Age", 18, 70, 30)
Income = st.number_input("Annual Income", 10000, 200000, 80000)
Credit_Score = st.number_input("Credit Score", 300, 900, 750)
DTI_Ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.25)
Loan_Amount = st.number_input("Loan Amount", 50000, 1000000, 300000)
Existing_Loans = st.number_input("Existing Loans", 0, 5, 0)
Dependents = st.number_input("Dependents", 0, 5, 1)
Collateral_Value = st.number_input("Collateral Value", 0, 2000000, 600000)

Employment_Status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
Education_Level = st.selectbox("Education Level", ["Graduate", "Postgraduate", "High School"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
Gender = st.selectbox("Gender", ["Male", "Female"])

# =========================
# Predict
# =========================
if st.button("Check Loan Approval"):

    input_df = pd.DataFrame([{
        "Age": Age,
        "Income": Income,
        "Credit_Score": Credit_Score,
        "DTI_Ratio": DTI_Ratio,
        "Loan_Amount": Loan_Amount,
        "Existing_Loans": Existing_Loans,
        "Dependents": Dependents,
        "Collateral_Value": Collateral_Value,
        "Employment_Status": Employment_Status,
        "Education_Level": Education_Level,
        "Property_Area": Property_Area,
        "Gender": Gender
    }])

    num_cols = input_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = input_df.select_dtypes(include=["object"]).columns.tolist()

    input_df[num_cols] = input_df[num_cols].fillna(0)
    input_df[cat_cols] = input_df[cat_cols].fillna("Unknown")

    encoded = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    final_df = pd.concat(
        [input_df[num_cols].reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    # 🔑 FORCE correct feature order
    final_df = final_df.reindex(columns=feature_order, fill_value=0)

    final_scaled = scaler.transform(final_df)

    proba = model.predict_proba(final_scaled)[0][1]

    st.write(f"Approval Probability: **{proba:.2f}**")

    if proba >= 0.5:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
