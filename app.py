
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('random_forest_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("Diabetes Prediction System")
st.subheader("Using SQL-Based Features: Risk Level & BMI Category")

st.markdown("---")
st.markdown("### Enter Patient Information:")

# Input fields
Pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0)
BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0)
SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0)
Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
Age = st.number_input("Age", min_value=1, max_value=120)

st.markdown("---")

# --- Handle missing / zero values (like in training) ---
# Replace 0 with median if needed (adjust medians based on your training set)
median_skin = 29  # example median from training data
median_insulin = 125  # example median from training data

SkinThickness = SkinThickness if SkinThickness > 0 else median_skin
Insulin = Insulin if Insulin > 0 else median_insulin

# --- Compute RiskLevel and BMICategory based on SQL logic ---
# RiskLevel (from Glucose)
if Glucose >= 140:
    RiskLevel = 2  # High
elif Glucose >= 100:
    RiskLevel = 1  # Medium
else:
    RiskLevel = 0  # Low

# BMICategory (from BMI)
if BMI < 18.5:
    BMICategory = 0  # Underweight
elif BMI < 25:
    BMICategory = 1  # Normal
elif BMI < 30:
    BMICategory = 2  # Overweight
else:
    BMICategory = 3  # Obese

# Predict Button
if st.button("Predict Diabetes"):
    # Prepare input for the model
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
               'RiskLevel', 'BMICategory']

    input_df = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DiabetesPedigreeFunction, Age,
                              RiskLevel, BMICategory]], columns=columns)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.error(f"Patient is likely **Diabetic** (Predicted: 1) \nProbability: {probability:.2%}")
    else:
        st.success(f"Patient is likely **Not Diabetic** (Predicted: 0) \nProbability: {probability:.2%}")

st.markdown("---")
st.caption("Created by Ramla Adan Yare · BSc Data Science")








