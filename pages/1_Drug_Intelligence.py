import streamlit as st
import pandas as pd
import joblib
import os
import ast

# Page config
st.set_page_config(page_title="Drug Intelligence Dashboard", page_icon="💊", layout="wide")

# Paths
DATA_PATH = os.path.join("Drug_module", "data", "Final_data", "drug_final_enriched_dataset.csv")
DEMAND_MODEL_PATH = os.path.join("Drug_module", "Models", "demand_model.pkl")
DEMAND_FEATURES_PATH = os.path.join("Drug_module", "Models", "demand_features.pkl")
EFFECTIVENESS_MODEL_PATH = os.path.join("Drug_module", "Models", "effectiveness_model.pkl")
EFFECTIVENESS_FEATURES_PATH = os.path.join("Drug_module", "Models", "effectiveness_features.pkl")

def fix_xgboost_model_compatibility(model):
    """
    Robust fix for XGBoost version mismatches after loading from joblib/pickle.
    Handles missing attributes like 'gpu_id', 'predictor', etc.
    """
    if hasattr(model, 'set_params'):
        # List of attributes that might be missing due to version differences
        potential_missing_attrs = {
            'gpu_id': None,
            'predictor': 'cpu_predictor',
            'base_score': 0.5,
            'n_estimators': 100,
            'importance_type': 'gain'
        }
        for attr, default_val in potential_missing_attrs.items():
            if not hasattr(model, attr):
                try:
                    setattr(model, attr, default_val)
                except Exception:
                    pass
    return model

# Functions
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    
    # Dynamic categorization logic
    def categorize_drug(row):
        if row['effectiveness_score'] > 8 and row['avg_sentiment'] > 0.3:
            return "High Performer"
        elif row['avg_sentiment'] < 0:
            return "High Risk"
        elif row['demand'] > df['demand'].quantile(0.75):
            return "High Demand"
        else:
            return "Moderate"
            
    df['drug_segment'] = df.apply(categorize_drug, axis=1)
    return df

@st.cache_resource
def load_demand_model():
    model = joblib.load(DEMAND_MODEL_PATH)
    features = joblib.load(DEMAND_FEATURES_PATH)
    model = fix_xgboost_model_compatibility(model)
    return model, features

@st.cache_resource
def load_effectiveness_model():
    model = joblib.load(EFFECTIVENESS_MODEL_PATH)
    features = joblib.load(EFFECTIVENESS_FEATURES_PATH)
    model = fix_xgboost_model_compatibility(model)
    return model, features

def predict_demand(row, model, features):
    try:
        # Create a deep copy to avoid modifying original row
        input_row = row.copy()
        
        # Ensure numeric conversion for required features
        numeric_cols = ['avg_sentiment', 'avg_rating', 'review_count', 'avg_usefulness', 
                        'effectiveness_score', 'sentiment_per_review', 'engagement_score', 
                        'sentiment_rating_proxy', 'sentiment_usefulness', 'rating_variation_proxy']
        
        for col in numeric_cols:
            if col in input_row:
                input_row[col] = pd.to_numeric(input_row[col], errors='coerce')
        
        input_df = pd.DataFrame([input_row])
        
        # One-hot encoding for categorical features (popularity)
        if 'popularity' in input_df.columns:
            pop_dummies = pd.get_dummies(input_df['popularity'], prefix='popularity')
            input_df = pd.concat([input_df.drop(columns=['popularity']), pop_dummies], axis=1)
        
        # Align columns using demand_features.pkl
        input_df = input_df.reindex(columns=features, fill_value=0)
        
        # Ensure all columns are float for XGBoost
        input_df = input_df.astype(float)
        
        prediction = model.predict(input_df)
        return max(0.0, float(prediction[0]))
    except Exception as e:
        st.error(f"Demand Prediction Error: {str(e)}")
        return 0.0

def predict_effectiveness(row, model, features):
    try:
        input_df = pd.DataFrame([row])
        # Align columns
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]
        
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Effectiveness Prediction Error: {str(e)}")
        return 0.0

def format_side_effects(side_effects_str):
    try:
        side_effects = ast.literal_eval(side_effects_str)
        if not side_effects:
            return "No significant side effects reported."
        return ", ".join([f"{effect} ({count})" for effect, count in side_effects])
    except:
        return side_effects_str

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

# Main Page
st.title("💊 Drug Intelligence Dashboard")
st.markdown("---")

df = load_data()

# Sample profiles fix - Using dynamic categories
st.subheader("Sample Drug Profiles")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

# Dynamic Sample Selection
high_perf_sample = df[df['drug_segment'] == "High Performer"].sample(1).iloc[0]['drugName'] if not df[df['drug_segment'] == "High Performer"].empty else df.iloc[0]['drugName']
risk_sample = df[df['drug_segment'] == "High Risk"].sample(1).iloc[0]['drugName'] if not df[df['drug_segment'] == "High Risk"].empty else df.iloc[1]['drugName']
demand_sample = df[df['drug_segment'] == "High Demand"].sample(1).iloc[0]['drugName'] if not df[df['drug_segment'] == "High Demand"].empty else df.iloc[2]['drugName']
mod_sample = df[df['drug_segment'] == "Moderate"].sample(1).iloc[0]['drugName'] if not df[df['drug_segment'] == "Moderate"].empty else df.iloc[3]['drugName']

selected_sample = None
if col_s1.button("High Performer Drug"): selected_sample = high_perf_sample
if col_s2.button("High Risk Drug"): selected_sample = risk_sample
if col_s3.button("High Demand Drug"): selected_sample = demand_sample
if col_s4.button("Moderate Drug"): selected_sample = mod_sample

# Selection
drug_list = sorted(df['drugName'].unique())
selected_drug = st.selectbox("Select Drug Name", drug_list, index=drug_list.index(selected_sample) if selected_sample in drug_list else 0)

if selected_drug:
    row = df[df['drugName'] == selected_drug].iloc[0]
    
    # Error Handling for predictions
    try:
        # Predict Demand
        demand_model, demand_features = load_demand_model()
        pred_demand = predict_demand(row, demand_model, demand_features)
        
        # Predict Effectiveness
        eff_model, eff_features = load_effectiveness_model()
        pred_effectiveness = predict_effectiveness(row, eff_model, eff_features)
        
        # Display Metrics
        st.container()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sentiment Score", f"{row['avg_sentiment']:.2f}")
        m2.metric("Effectiveness Score", f"{pred_effectiveness:.2f}")
        m3.metric("Drug Segment", row['drug_segment'])
        m4.metric("Predicted Demand", f"{pred_demand:.2f}")
        
        st.markdown("---")
        
        # Side Effects
        st.subheader("Reported Side Effects")
        st.info(format_side_effects(row['top_side_effects']))
        
        # Additional data
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Insights")
            insights = ast.literal_eval(row['insights']) if isinstance(row['insights'], str) else row['insights']
            if insights:
                for ins in insights:
                    st.warning(ins)
            else:
                st.write("No specific insights available.")
        with c2:
            st.write("### Recommended Actions")
            actions = ast.literal_eval(row['actions']) if isinstance(row['actions'], str) else row['actions']
            if actions:
                for act in actions:
                    st.success(act)
            else:
                st.write("No specific actions recommended.")
                
    except Exception as e:
        st.error(f"An error occurred while processing the drug data: {str(e)}")
        st.info("Ensure the following files exist:\n\nDrug Module:\n* demand_model.pkl\n* demand_features.pkl\n* effectiveness_model.pkl")

st.markdown("---")
st.markdown("Built as part of Pharma Decision Intelligence Platform")
