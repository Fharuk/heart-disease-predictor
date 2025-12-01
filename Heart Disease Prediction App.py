import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# -------------------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# -------------------------------------------------------------------------------------------------
# CACHED RESOURCE LOADING (The Performance Fix)
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Load models and scalers once and cache them.
    Returns None for missing files to prevent app crash.
    """
    artifacts = {}
    files = {
        "rf_model": "rf_model_subset_A.pkl",
        "scaler_A": "scaler_subset_A.pkl",
        "xgb_model": "xgb_model_subset_B.pkl",
        "scaler_B": "scaler_subset_B.pkl"
    }

    for name, path in files.items():
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    artifacts[name] = pickle.load(f)
            except Exception as e:
                st.error(f"⚠️ Error loading {path}: {e}")
                artifacts[name] = None
        else:
            st.error(f"⚠️ File not found: {path}")
            artifacts[name] = None
            
    return artifacts

# Load everything once
artifacts = load_artifacts()

# -------------------------------------------------------------------------------------------------
# UI & INPUTS
# -------------------------------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predicts likelihood using **Random Forest (Subset A)** and **XGBoost (Subset B)**.")

# Feature Definitions
features_A = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
              "restecg", "thalach", "exang", "oldpeak", "slope", 
              "ca", "thal", "smoking", "diabetes", "bmi"]

features_B = ["age", "sex", "cp", "trestbps", "chol", 
              "thalach", "exang", "oldpeak", "slope", 
              "ca", "thal", "smoking", "diabetes", "bmi"]

st.sidebar.header("📝 Patient Vitals")

# Collect inputs
input_dict = {
    "age": st.sidebar.number_input("Age", 18, 100, 45),
    "sex": st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female"),
    "cp": st.sidebar.slider("Chest Pain Type", 0, 3, 1),
    "trestbps": st.sidebar.number_input("Resting BP", 80, 200, 120),
    "chol": st.sidebar.number_input("Cholesterol", 100, 400, 200),
    "fbs": st.sidebar.selectbox("Fasting BS > 120", [0, 1]),
    "restecg": st.sidebar.slider("Resting ECG", 0, 2, 1),
    "thalach": st.sidebar.number_input("Max Heart Rate", 60, 220, 150),
    "exang": st.sidebar.selectbox("Exercise Angina", [0, 1]),
    "oldpeak": st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1),
    "slope": st.sidebar.slider("Slope", 0, 2, 1),
    "ca": st.sidebar.slider("Major Vessels (0-4)", 0, 4, 0),
    "thal": st.sidebar.slider("Thalassemia (0-3)", 0, 3, 2),
    "smoking": st.sidebar.selectbox("Smoking", [0, 1]),
    "diabetes": st.sidebar.selectbox("Diabetes", [0, 1]),
    "bmi": st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
}

input_df = pd.DataFrame([input_dict])

# -------------------------------------------------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------------------------------------------------
if st.button("🔍 Analyze Risk"):
    st.subheader("Results")
    
    # 1. Random Forest Prediction
    rf_model = artifacts["rf_model"]
    scaler_A = artifacts["scaler_A"]
    
    if rf_model and scaler_A:
        # Align features for A
        input_A = input_df.reindex(columns=features_A, fill_value=0)
        try:
            scaled_A = scaler_A.transform(input_A)
            prob_A = rf_model.predict_proba(scaled_A)[0][1]
            pred_A = rf_model.predict(scaled_A)[0]
            
            color = "red" if pred_A == 1 else "green"
            st.markdown(f"**Random Forest:** :{color}[{'Heart Disease' if pred_A==1 else 'Normal'}] (Risk: {prob_A:.2%})")
        except Exception as e:
            st.error(f"RF Error: {e}")

    # 2. XGBoost Prediction
    xgb_model = artifacts["xgb_model"]
    scaler_B = artifacts["scaler_B"]
    
    if xgb_model and scaler_B:
        # Align features for B
        input_B = input_df.reindex(columns=features_B, fill_value=0)
        try:
            scaled_B = scaler_B.transform(input_B)
            prob_B = xgb_model.predict_proba(scaled_B)[0][1]
            pred_B = xgb_model.predict(scaled_B)[0]
            
            color = "red" if pred_B == 1 else "green"
            st.markdown(f"**XGBoost:** :{color}[{'Heart Disease' if pred_B==1 else 'Normal'}] (Risk: {prob_B:.2%})")
        except Exception as e:
            st.error(f"XGB Error: {e}")