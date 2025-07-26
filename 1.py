import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Create and Train Model
# -------------------------------
@st.cache_resource
def train_and_save_model():
    # Sample dataset
    data = {
        "Gender": [1, 0, 1, 0, 1],
        "Married": [1, 0, 1, 0, 1],
        "Education": [0, 1, 0, 1, 0],
        "Self_Employed": [0, 1, 0, 1, 0],
        "ApplicantIncome": [5000, 3000, 4000, 2500, 6000],
        "CoapplicantIncome": [0, 1500, 1200, 1800, 0],
        "LoanAmount": [200, 100, 150, 90, 220],
        "Loan_Amount_Term": [360, 360, 360, 360, 360],
        "Credit_History": [1.0, 0.0, 1.0, 0.0, 1.0],
        "Property_Area": [2, 0, 1, 0, 2],
        "Loan_Status": [1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

# Load model (train it if not exists)
model = train_and_save_model()

# -------------------------------
# Step 2: Streamlit UI
# -------------------------------

# Optional background styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("it.jpg");
        background-size: cover;
        background-position: center;
        color: gold;
        font-family: 'Segoe UI', sans-serif;
    }}
    .css-1d391kg p, .css-1d391kg label {{
        font-size: 18px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Loan Prediction Portal")
st.subheader("By Smart Finance System")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encoding
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Predict on button click
if st.button("Check Loan Eligibility"):
    input_data = np.array([[gender, married, education, self_employed,
                            applicant_income, coapplicant_income,
                            loan_amount, loan_amount_term,
                            credit_history, property_area]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ You are eligible for the loan.")
    else:
        st.error("❌ Sorry, you are not eligible for the loan.")
