import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import xgboost as xgb

# Page config
st.set_page_config(page_title="Demo Trials", page_icon="📂", layout="wide")

# Paths
MODEL_PATH = "Clinical_Trial_module/Models/clinical/model.json"
COLS_PATH = "Clinical_Trial_module/Models/clinical/columns.pkl"

# Functions
@st.cache_resource
def load_clinical_models():
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        with open(COLS_PATH, "rb") as f:
            columns = pickle.load(f)
        return model, list(columns)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def predict_trial_success(inputs, model, columns):
    try:
        input_data = {
            "phase": [int(inputs['phase'])],
            "log_enrollment": [np.log1p(float(inputs['enrollment']))],
            "duration_months": [float(inputs['duration'])],
            "study_type": [inputs['study_type'].upper()],
            "funder_type": [inputs['funder_type'].upper()]
        }
        
        df_input = pd.DataFrame(input_data)
        df_encoded = pd.get_dummies(df_input)
        df_final = df_encoded.reindex(columns=columns, fill_value=0)
        
        prediction = model.predict_proba(df_final)[0][1]
        return float(prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0

# Sidebar navigation
st.sidebar.title("PHARMA AI")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Home", icon="🏠")

st.sidebar.subheader("💊 Intelligence")
st.sidebar.page_link("pages/1_Drug_Intelligence.py", label="Drug Overview", icon="📋")
st.sidebar.page_link("pages/2_Patient_Insights.py", label="Patient Insights", icon="💬")
st.sidebar.page_link("pages/4_Decision_Intelligence.py", label="Decision Panel", icon="🧠")

st.sidebar.subheader("🧪 Trials")
st.sidebar.page_link("pages/5_Predict_Trial.py", label="Predict Success", icon="🔮")
st.sidebar.page_link("pages/6_Demo_Trials.py", label="Demo Scenarios", icon="📂")
st.sidebar.page_link("pages/7_About_Model.py", label="Technical Docs", icon="ℹ️")

# Main Page
st.title("📂 Clinical Trial Demo Scenarios")
st.markdown("---")

scenarios = [
    {
        "name": "Oncology - Phase 1 High Risk",
        "phase": 1,
        "study_type": "INTERVENTIONAL",
        "enrollment": 15,
        "duration": 24, # 2 years
        "funder_type": "INDUSTRY",
        "desc": "Testing a novel CDK inhibitor for advanced solid tumors."
    },
    {
        "name": "Cardiovascular - Phase 3 Low Risk",
        "phase": 3,
        "study_type": "INTERVENTIONAL",
        "enrollment": 5000,
        "duration": 36, # 3 years
        "funder_type": "INDUSTRY",
        "desc": "Beta-blocker vs Placebo for Chronic Heart Failure."
    },
    {
        "name": "Endocrinology - Phase 2 Medium Risk",
        "phase": 2,
        "study_type": "INTERVENTIONAL",
        "enrollment": 150,
        "duration": 12, # 1 year
        "funder_type": "INDUSTRY",
        "desc": "GLP-1 Analog testing for Type 2 Diabetes management."
    },
    {
        "name": "Neurology - Phase 1 High Risk",
        "phase": 1,
        "study_type": "INTERVENTIONAL",
        "enrollment": 24,
        "duration": 18, # 1.5 years
        "funder_type": "INDUSTRY",
        "desc": "Amyloid-beta antibody for Alzheimer's Disease treatment."
    },
    {
        "name": "Infectious Disease - Phase 3 Low Risk",
        "phase": 3,
        "study_type": "INTERVENTIONAL",
        "enrollment": 30000,
        "duration": 6, # 6 months
        "funder_type": "INDUSTRY",
        "desc": "Quadrivalent Vaccine for Seasonal Influenza."
    },
    {
        "name": "Respiratory - Phase 2 Medium Risk",
        "phase": 2,
        "study_type": "INTERVENTIONAL",
        "enrollment": 200,
        "duration": 9, # 9 months
        "funder_type": "INDUSTRY",
        "desc": "Inhaled Corticosteroid for Severe Asthma symptoms."
    },
    {
        "name": "Dermatology - Phase 2 Low Risk",
        "phase": 2,
        "study_type": "INTERVENTIONAL",
        "enrollment": 300,
        "duration": 6, # 6 months
        "funder_type": "INDUSTRY",
        "desc": "Topical JAK Inhibitor for Psoriasis treatment."
    },
    {
        "name": "Immunology - Phase 1 High Risk",
        "phase": 1,
        "study_type": "INTERVENTIONAL",
        "enrollment": 30,
        "duration": 12, # 1 year
        "funder_type": "INDUSTRY",
        "desc": "CAR-T Therapy for Rheumatoid Arthritis."
    },
    {
        "name": "Ophthalmology - Phase 3 Medium Risk",
        "phase": 3,
        "study_type": "INTERVENTIONAL",
        "enrollment": 800,
        "duration": 24, # 2 years
        "funder_type": "INDUSTRY",
        "desc": "Anti-VEGF Injection for Macular Degeneration."
    },
    {
        "name": "Rare Disease - Phase 2 High Risk",
        "phase": 2,
        "study_type": "INTERVENTIONAL",
        "enrollment": 40,
        "duration": 36, # 3 years
        "funder_type": "INDUSTRY",
        "desc": "Exon-skipping ASO for Duchenne Muscular Dystrophy."
    }
]

model, columns = load_clinical_models()

if model is not None:
    st.write("Select a pre-configured demo scenario to analyze its predicted success probability.")
    
    cols = st.columns(2)
    for i, sc in enumerate(scenarios):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"### {sc['name']}")
                st.info(f"**Description:** {sc['desc']}")
                
                # Input metrics display
                m1, m2, m3 = st.columns(3)
                m1.metric("Phase", sc['phase'])
                m2.metric("Enrollment", sc['enrollment'])
                m3.metric("Duration", f"{sc['duration']}mo")
                
                m4, m5 = st.columns(2)
                m4.metric("Type", sc['study_type'].capitalize())
                m5.metric("Funder", sc['funder_type'].capitalize())
                
                if st.button(f"Analyze {sc['name']}", key=f"btn_{i}", use_container_width=True):
                    inputs = {
                        'phase': sc['phase'],
                        'enrollment': sc['enrollment'],
                        'duration': sc['duration'],
                        'study_type': sc['study_type'],
                        'funder_type': sc['funder_type']
                    }
                    prob = predict_trial_success(inputs, model, columns)
                    
                    st.markdown("---")
                    if prob > 0.7:
                        st.success(f"### Success Probability: {prob:.2%}")
                        st.write("✅ **Recommendation:** This trial shows high success potential. Proceed with resource allocation.")
                    elif prob >= 0.4:
                        st.warning(f"### Success Probability: {prob:.2%}")
                        st.write("⚠️ **Recommendation:** Moderate risk detected. Monitor trial protocols closely.")
                    else:
                        st.error(f"### Success Probability: {prob:.2%}")
                        st.write("❌ **Recommendation:** High risk of failure. Review study design or intervention strategy.")
                
                st.markdown("<br>", unsafe_allow_html=True)
else:
    st.error("Clinical models not found. Please check configuration.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Built as part of the Pharma Decision Intelligence Platform | © 2026</p>", unsafe_allow_html=True)
