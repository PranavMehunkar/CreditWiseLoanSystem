import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="CreditWise Loan Approval", page_icon="💰")
st.title("💰 CreditWise Loan Approval Prediction")

# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    try:
        with open("logistic_model.pkl", "rb") as f:
            logistic_model = pickle.load(f)
        with open("knn_model.pkl", "rb") as f:
            knn_model = pickle.load(f)
        with open("nb_model.pkl", "rb") as f:
            nb_model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("edu_le.pkl", "rb") as f:
            le_edu = pickle.load(f)
        with open("feature_order.pkl", "rb") as f:
            feature_order = pickle.load(f)
        return logistic_model, knn_model, nb_model, scaler, encoder, le_edu, feature_order
    except FileNotFoundError:
        st.error("⚠️ Some artifact files are missing! Upload all .pkl files.")
        st.stop()

logistic_model, knn_model, nb_model, scaler, encoder, le_edu, feature_order = load_artifacts()

# =========================
# Model selector
# =========================
model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "KNN", "Naive Bayes"])
if model_choice == "Logistic Regression":
    model = logistic_model
elif model_choice == "KNN":
    model = knn_model
else:
    model = nb_model

# =========================
# User Input
# =========================
st.subheader("Enter Applicant Details:")

# Numeric inputs
Age = st.number_input("Age", 18, 100, 30)
Applicant_Income = st.number_input("Applicant Income", 0, 1000000, 50000)
Coapplicant_Income = st.number_input("Coapplicant Income", 0, 1000000, 0)
Credit_Score = st.number_input("Credit Score", 300, 900, 650)
DTI_Ratio = st.number_input("Debt-to-Income Ratio", 0.0, 100.0, 20.0)
Savings = st.number_input("Savings", 0, 1000000, 10000)
Existing_Loans = st.number_input("Existing Loans", 0, 100, 0)
Collateral_Value = st.number_input("Collateral Value", 0, 1000000, 20000)
Dependents = st.number_input("Number of Dependents", 0, 10, 0)
Loan_Amount = st.number_input("Loan Amount", 0, 1000000, 20000)

# Categorical inputs
Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
Employment_Status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Loan_Purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Other"])
Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Employer_Category = st.selectbox("Employer Category", ["Private", "Government", "Other"])

# =========================
# Create DataFrame
# =========================
input_dict = {
    "Age": Age,
    "Applicant_Income": Applicant_Income,
    "Coapplicant_Income": Coapplicant_Income,
    "Credit_Score": Credit_Score,
    "DTI_Ratio": DTI_Ratio,
    "Savings": Savings,
    "Existing_Loans": Existing_Loans,
    "Collateral_Value": Collateral_Value,
    "Dependents": Dependents,
    "Loan_Amount": Loan_Amount,
    "Education_Level": le_edu.transform([Education_Level])[0],
    "Employment_Status": Employment_Status,
    "Marital_Status": Marital_Status,
    "Loan_Purpose": Loan_Purpose,
    "Property_Area": Property_Area,
    "Gender": Gender,
    "Employer_Category": Employer_Category
}

input_df = pd.DataFrame([input_dict])

# =========================
# Encode categorical columns (excluding Education_Level)
# =========================
cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
            "Property_Area","Gender","Employer_Category"]

encoded = encoder.transform(input_df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
final_df = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)

# =========================
# Add missing columns if any
# =========================
missing_cols = set(feature_order) - set(final_df.columns)
for col in missing_cols:
    final_df[col] = 0

final_df = final_df[feature_order]

# =========================
# Scale features
# =========================
final_scaled = scaler.transform(final_df)

# =========================
# Predict
# =========================
if st.button("Predict Loan Approval"):
    pred = model.predict(final_scaled)[0]
    proba = model.predict_proba(final_scaled)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result:")
    if pred == 1:
        st.success("✅ Loan will be Approved")
    else:
        st.error("❌ Loan will NOT be Approved")

    if proba is not None:
        st.info(f"Prediction Probability: {proba*100:.2f}%")
