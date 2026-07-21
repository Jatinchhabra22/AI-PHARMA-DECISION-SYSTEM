import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page config
st.set_page_config(page_title="Predict Trial Success", page_icon="🔮", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Home", icon="🏠")

st.sidebar.subheader("💊 Drug Intelligence")
st.sidebar.page_link("pages/1_Drug_Intelligence.py", label="Drug Overview", icon="📋")
st.sidebar.page_link("pages/2_Patient_Insights.py", label="Patient Insights", icon="💬")
st.sidebar.page_link("pages/4_Decision_Intelligence.py", label="Decision Panel", icon="🧠")

st.sidebar.subheader("🧪 Clinical Trials")
st.sidebar.page_link("pages/5_Predict_Trial.py", label="Predict Trial", icon="🔮")
st.sidebar.page_link("pages/6_Demo_Trials.py", label="Demo Trials", icon="📂")
st.sidebar.page_link("pages/7_About_Model.py", label="About Model", icon="ℹ️")

# -------------------------
# LOAD MODEL
# -------------------------
import xgboost as xgb

# LOAD MODEL (JSON WAY)
model = xgb.XGBClassifier()
try:
    model.load_model("Clinical_Trial_module/Models/clinical/model.json")
except Exception:
    st.error("Model file not found!")

# LOAD COLUMNS
try:
    columns = pickle.load(open("Clinical_Trial_module/Models/clinical/columns.pkl", "rb"))
    columns = list(columns)
except Exception:
    st.error("Columns file not found!")
    columns = []

st.title("🔮 Clinical Trial Success Predictor")
st.markdown("---")

st.write("Enter the trial parameters to estimate its success probability based on historical data.")


# -------------------------
# INPUTS
# -------------------------
phase = st.selectbox("Phase", [1, 2, 3])
enrollment = st.number_input("Enrollment", min_value=10, value=100)
duration = st.number_input("Duration (months)", min_value=1, value=12)

study_type = st.selectbox("Study Type", ["INTERVENTIONAL", "OBSERVATIONAL"])
funder_type = st.selectbox("Funder Type", ["INDUSTRY", "OTHER"])

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([{
        "phase": phase,
        "log_enrollment": np.log1p(enrollment),
        "duration_months": duration,
        "study_type": study_type,
        "funder_type": funder_type
    }])

    # encoding
    input_df = pd.get_dummies(input_df)

    # align columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # predict
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Success Probability: {prob:.2f}")

    if prob > 0.7:
        st.success("Low Risk Trial")
    elif prob > 0.4:
        st.warning("Medium Risk Trial")
    else:
        st.error("High Risk Trial")