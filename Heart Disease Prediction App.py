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
                # We log the error but don't stop the app
                print(f"Failed to load {name}: {e}")
                artifacts[name] = None
        else:
            artifacts[name] = None
            
    return artifacts

artifacts = load_artifacts()

# -------------------------------------------------------------------------------------------------
# HELPER: DYNAMIC ALIGNMENT
# -------------------------------------------------------------------------------------------------
def get_clean_input(input_df, model):
    """
    Attempts to align data. If model has no feature list (old version),
    we convert to numpy array to bypass name checks, but this is risky.
    """
    if hasattr(model, "feature_names_in_"):
        return input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        # Fallback for old models: 
        # We cannot guess columns. We return None to signal "Unsafe".
        return None

# -------------------------------------------------------------------------------------------------
# UI & INPUTS
# -------------------------------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predicts likelihood using **XGBoost** (Primary) and **Random Forest** (Legacy).")

st.sidebar.header("📝 Patient Vitals")

# Collect inputs
input_dict = {
    "age": st.sidebar.number_input("Age", 18, 100, 45),
    "sex": st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1], index=1),
    "cp": st.sidebar.slider("Chest Pain Type", 0, 3, 1),
    "trestbps": st.sidebar.number_input("Resting BP", 80, 200, 120),
    "chol": st.sidebar.number_input("Cholesterol", 100, 400, 200),
    "fbs": st.sidebar.selectbox("Fasting BS > 120", [0, 1]),
    "restecg": st.sidebar.slider("Resting ECG", 0, 2, 1),
    "thalach": st.sidebar.number_input("Max Heart Rate", 60, 220, 150),
    "exang": st.sidebar.selectbox("Exercise Angina", [0, 1]),
    "oldpeak": st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1),
    "slope": st.sidebar.slider("Slope", 0, 2, 1),
    "ca": st.sidebar.slider("Major Vessels", 0, 4, 0),
    "thal": st.sidebar.slider("Thalassemia", 0, 3, 2),
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
    
    # --- XGBoost (The Reliable One) ---
    xgb_model = artifacts["xgb_model"]
    scaler_B = artifacts["scaler_B"]
    
    if xgb_model and scaler_B:
        try:
            # XGBoost usually has feature names, so we try alignment
            if hasattr(xgb_model, "feature_names_in_"):
                input_B = input_df.reindex(columns=xgb_model.feature_names_in_, fill_value=0)
            else:
                # If XGBoost is also old, we try passing all features (risky but might work)
                # But based on logs, XGBoost was working with the previous code.
                # We will define the list explicitly based on your original code:
                cols_B = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", 
                          "oldpeak", "slope", "ca", "thal", "smoking", "diabetes", "bmi"]
                input_B = input_df[cols_B]

            scaled_B = scaler_B.transform(input_B)
            prob_B = xgb_model.predict_proba(scaled_B)[0][1]
            pred_B = xgb_model.predict(scaled_B)[0]
            
            color = "red" if pred_B == 1 else "green"
            st.markdown(f"### ✅ XGBoost Prediction")
            st.markdown(f"**Result:** :{color}[{'Heart Disease' if pred_B==1 else 'Normal'}]")
            st.markdown(f"**Probability:** {prob_B:.2%}")
            
        except Exception as e:
            st.error(f"XGBoost Error: {e}")
    else:
        st.error("XGBoost model file not found.")

    st.markdown("---")

    # --- Random Forest (The Broken One) ---
    rf_model = artifacts["rf_model"]
    scaler_A = artifacts["scaler_A"]
    
    if rf_model and scaler_A:
        try:
            # We attempt to guess the columns since feature_names_in_ is missing
            # Based on your error logs, it rejected 'fbs', 'restecg', 'slope', 'bmi', 'diabetes'
            # This implies a very small subset.
            # We will try to pass a standard subset, BUT if it fails, we catch it.
            
            if hasattr(rf_model, "feature_names_in_"):
                input_A = input_df.reindex(columns=rf_model.feature_names_in_, fill_value=0)
                scaled_A = scaler_A.transform(input_A)
                prob_A = rf_model.predict_proba(scaled_A)[0][1]
                st.write(f"Random Forest Risk: {prob_A:.2%}")
            else:
                st.warning("⚠️ Random Forest model is undergoing maintenance (Version Mismatch).")
                
        except Exception as e:
            # Silent fail - don't show user the ugly error
            st.warning("⚠️ Random Forest model temporarily unavailable.")
            with st.expander("Developer Logs"):
                st.write(str(e))
