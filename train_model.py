#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Load dataset
# =========================
df = pd.read_csv("data/loan_approval_data.csv")

# =========================
# Target column
# =========================
TARGET = "Loan_Approved"

# Drop ID if exists
if "Applicant_ID" in df.columns:
    df.drop(columns=["Applicant_ID"], inplace=True)

# =========================
# Identify numeric and categorical columns
# =========================
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove target from categorical columns
if TARGET in cat_cols:
    cat_cols.remove(TARGET)

# =========================
# Handle missing values
# =========================
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# =========================
# Label Encode Education_Level
# =========================
le_edu = LabelEncoder()
if "Education_Level" in df.columns:
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

# =========================
# Encode target
# =========================
le_target = LabelEncoder()
df[TARGET] = le_target.fit_transform(df[TARGET])

# =========================
# OneHot Encode remaining categorical columns
# =========================
ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
            "Property_Area","Gender","Employer_Category"]

encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(ohe_cols), index=df.index)

# =========================
# Combine features
# =========================
X = pd.concat([df[num_cols], df["Education_Level"], encoded_df], axis=1)
y = df[TARGET]

feature_order = X.columns.tolist()

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Scale numeric features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Train models
# =========================
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB()
}

print("\n📊 Model Performance\n" + "-"*30)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print(f"\n{name.upper()} MODEL")
    print("Accuracy :", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='weighted'))
    print("Recall   :", recall_score(y_test, preds, average='weighted'))
    print("F1 Score :", f1_score(y_test, preds, average='weighted'))

# =========================
# Save artifacts
# =========================
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("edu_le.pkl", "wb") as f:
    pickle.dump(le_edu, f)

with open("feature_order.pkl", "wb") as f:
    pickle.dump(feature_order, f)

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(models["logistic"], f)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(models["knn"], f)

with open("nb_model.pkl", "wb") as f:
    pickle.dump(models["naive_bayes"], f)

print("\n✅ Training complete. All artifacts saved.")


# In[ ]:
