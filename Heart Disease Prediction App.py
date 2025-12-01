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
    """
    Load only the functioning XGBoost model and scaler.
    """
    artifacts = {}
    files = {
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
            st.error(f"⚠️ Critical File Missing: {path}")
            artifacts[name] = None
            
    return artifacts

artifacts = load_artifacts()

# -------------------------------------------------------------------------------------------------
# UI & INPUTS
# -------------------------------------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Assessment")
st.markdown("Professional assessment tool powered by **XGBoost (Gradient Boosting)**.")

with st.form("patient_data"):
    st.subheader("Patient Vitals")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 100, 45)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.slider("Chest Pain Type", 0, 3, 1, help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting BS > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
        restecg = st.slider("Resting ECG Results", 0, 2, 1)

    with col2:
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        slope = st.slider("Slope of Peak Exercise", 0, 2, 1)
        ca = st.slider("Major Vessels Colored (0-4)", 0, 4, 0)
        thal = st.slider("Thalassemia", 0, 3, 2, help="0: Normal, 1: Fixed Defect, 2: Reversable Defect")
        
    st.subheader("Lifestyle Factors")
    col3, col4, col5 = st.columns(3)
    with col3:
        smoking = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col4:
        diabetes = st.selectbox("Diabetic", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col5:
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0, step=0.1)

    submit_button = st.form_submit_button("🔍 Run Analysis", type="primary")

# -------------------------------------------------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------------------------------------------------
if submit_button:
    xgb_model = artifacts["xgb_model"]
    scaler_B = artifacts["scaler_B"]
    
    if xgb_model and scaler_B:
        try:
            # Prepare Input Dictionary
            input_dict = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
                "smoking": smoking, "diabetes": diabetes, "bmi": bmi
            }
            input_df = pd.DataFrame([input_dict])

            # Align Features for XGBoost
            # If model has saved features, use them. If not, use the standard Subset B list.
            if hasattr(xgb_model, "feature_names_in_"):
                aligned_df = input_df.reindex(columns=xgb_model.feature_names_in_, fill_value=0)
            else:
                # Fallback based on your successful test earlier
                cols_B = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", 
                          "oldpeak", "slope", "ca", "thal", "smoking", "diabetes", "bmi"]
                aligned_df = input_df[cols_B]

            # Scale and Predict
            scaled_input = scaler_B.transform(aligned_df)
            prediction = xgb_model.predict(scaled_input)[0]
            probability = xgb_model.predict_proba(scaled_input)[0][1]

            # Display Results
            st.divider()
            st.subheader("Assessment Result")
            
            if prediction == 1:
                st.error(f"### High Risk Detected")
                st.markdown(f"**Probability:** {probability:.1%}")
                st.warning("Please consult a cardiologist for further evaluation.")
            else:
                st.success(f"### Low Risk Detected")
                st.markdown(f"**Probability:** {probability:.1%}")
                st.info("Maintain a healthy lifestyle to keep risk low.")

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            st.info("Please verify that all inputs are within valid ranges.")
