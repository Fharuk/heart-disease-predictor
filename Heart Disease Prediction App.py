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
# CACHED RESOURCE LOADING
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
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

artifacts = load_artifacts()

# -------------------------------------------------------------------------------------------------
# DYNAMIC ALIGNMENT FUNCTION (The Fix)
# -------------------------------------------------------------------------------------------------
def align_data_to_model(input_df, model):
    """
    Automatically filters and sorts the input dataframe to match 
    EXACTLY what the model expects.
    """
    try:
        # Get the feature names the model was trained on
        expected_features = model.feature_names_in_
        
        # 1. Keep only columns that exist in expected_features
        # 2. Add missing columns filling with 0
        # 3. Reorder to match the model's order
        aligned_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        return aligned_df, expected_features
    except AttributeError:
        st.error("Model does not have feature_names_in_. It might be an old version.")
        return input_df, []

# -------------------------------------------------------------------------------------------------
# UI & INPUTS
# -------------------------------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predicts likelihood using **Random Forest** and **XGBoost**.")

st.sidebar.header("📝 Patient Vitals")

# Collect inputs (We collect EVERYTHING, let the adapter sort it out later)
input_dict = {
    "age": st.sidebar.number_input("Age", 18, 100, 45),
    "sex": st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1], index=1),
    "cp": st.sidebar.slider("Chest Pain Type (cp)", 0, 3, 1),
    "trestbps": st.sidebar.number_input("Resting BP (trestbps)", 80, 200, 120),
    "chol": st.sidebar.number_input("Cholesterol (chol)", 100, 400, 200),
    "fbs": st.sidebar.selectbox("Fasting BS > 120 (fbs)", [0, 1]),
    "restecg": st.sidebar.slider("Resting ECG (restecg)", 0, 2, 1),
    "thalach": st.sidebar.number_input("Max Heart Rate (thalach)", 60, 220, 150),
    "exang": st.sidebar.selectbox("Exercise Angina (exang)", [0, 1]),
    "oldpeak": st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1),
    "slope": st.sidebar.slider("Slope", 0, 2, 1),
    "ca": st.sidebar.slider("Major Vessels (ca)", 0, 4, 0),
    "thal": st.sidebar.slider("Thalassemia (thal)", 0, 3, 2),
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
    
    # --- Random Forest ---
    rf_model = artifacts["rf_model"]
    scaler_A = artifacts["scaler_A"]
    
    if rf_model and scaler_A:
        try:
            # ALIGNMENT MAGIC
            input_A, features_A = align_data_to_model(input_df, rf_model)
            
            # Scale
            scaled_A = scaler_A.transform(input_A)
            
            # Predict
            prob_A = rf_model.predict_proba(scaled_A)[0][1]
            pred_A = rf_model.predict(scaled_A)[0]
            
            color = "red" if pred_A == 1 else "green"
            st.markdown(f"**Random Forest:** :{color}[{'Heart Disease' if pred_A==1 else 'Normal'}] (Risk: {prob_A:.2%})")
            
            with st.expander("See RF Features Used"):
                st.write(list(features_A))
                
        except Exception as e:
            st.error(f"RF Error: {e}")

    # --- XGBoost ---
    xgb_model = artifacts["xgb_model"]
    scaler_B = artifacts["scaler_B"]
    
    if xgb_model and scaler_B:
        try:
            # ALIGNMENT MAGIC
            input_B, features_B = align_data_to_model(input_df, xgb_model)
            
            # Scale
            scaled_B = scaler_B.transform(input_B)
            
            # Predict
            prob_B = xgb_model.predict_proba(scaled_B)[0][1]
            pred_B = xgb_model.predict(scaled_B)[0]
            
            color = "red" if pred_B == 1 else "green"
            st.markdown(f"**XGBoost:** :{color}[{'Heart Disease' if pred_B==1 else 'Normal'}] (Risk: {prob_B:.2%})")
            
            with st.expander("See XGB Features Used"):
                st.write(list(features_B))

        except Exception as e:
            st.error(f"XGB Error: {e}")
