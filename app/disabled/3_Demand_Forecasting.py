import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Demand Prediction & Trends", page_icon="📈", layout="wide")

# Paths
DATA_PATH = os.path.join("Drug_module", "data", "Final_data", "drug_final_enriched_dataset.csv")
DEMAND_MODEL_PATH = os.path.join("Drug_module", "Models", "demand_model.pkl")
DEMAND_FEATURES_PATH = os.path.join("Drug_module", "Models", "demand_features.pkl")
LSTM_MODEL_PATH = os.path.join("Drug_module", "Models", "lstm_demand_model.h5")
LSTM_SCALER_PATH = os.path.join("Drug_module", "Models", "lstm_scaler.pkl")

# Functions
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

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

@st.cache_resource
def load_ml_models():
    try:
        model = joblib.load(DEMAND_MODEL_PATH)
        features = joblib.load(DEMAND_FEATURES_PATH)
        model = fix_xgboost_model_compatibility(model)
        return model, features
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None, None

@st.cache_resource
def load_dl_models():
    try:
        model = load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(LSTM_SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading DL models: {e}")
        return None, None

def predict_demand_ml(row, model, features):
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
            # Handle case where popularity might be 'high' but model expects 'low'/'medium'
            pop_dummies = pd.get_dummies(input_df['popularity'], prefix='popularity')
            input_df = pd.concat([input_df.drop(columns=['popularity']), pop_dummies], axis=1)
        
        # Align columns using demand_features.pkl
        input_df = input_df.reindex(columns=features, fill_value=0)
        
        # Ensure all columns are float for XGBoost
        input_df = input_df.astype(float)
        
        prediction = model.predict(input_df)
        return max(0.0, float(prediction[0]))
    except Exception as e:
        st.error(f"ML Prediction Error: {e}")
        return 0.0

def run_lstm_forecast(current_demand, model, scaler):
    try:
        # LSTM model expects (samples, timesteps, features)
        # We use a sequence length of 10
        sequence_length = 10
        
        # Create a synthetic historical trend for visualization
        # In a real app, this would be actual historical data
        historical_data = np.linspace(current_demand * 0.9, current_demand, sequence_length).reshape(-1, 1)
        
        # Scale the data using the loaded scaler
        scaled_data = scaler.transform(historical_data)
        
        # Reshape to (1, 10, 1)
        input_seq = scaled_data.reshape(1, sequence_length, 1)
        
        # Forecast next 5 time steps
        forecast = []
        current_seq = input_seq.copy()
        
        for _ in range(5):
            # Predict next step
            try:
                next_val = model.predict(current_seq, verbose=0)
                forecast.append(next_val[0, 0])
                
                # Update sequence for next prediction (sliding window)
                current_seq = np.roll(current_seq, -1, axis=1)
                current_seq[0, -1, 0] = next_val[0, 0]
            except Exception as dl_err:
                st.warning(f"DL Step prediction warning: {dl_err}")
                forecast.append(current_seq[0, -1, 0] * 1.01) # fallback: 1% growth
            
        # Inverse transform the forecast back to original scale
        forecast_array = np.array(forecast).reshape(-1, 1)
        forecast_unscaled = scaler.inverse_transform(forecast_array).flatten()
        
        return historical_data.flatten(), forecast_unscaled
    except Exception as e:
        st.error(f"LSTM Forecasting Error: {e}")
        # Return fallback data so the chart doesn't crash
        return np.linspace(current_demand*0.9, current_demand, 10), np.linspace(current_demand, current_demand*1.1, 5)

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Home", icon="🏠")

st.sidebar.subheader("💊 Drug Intelligence")
st.sidebar.page_link("pages/1_Drug_Intelligence.py", label="Drug Overview", icon="📋")
st.sidebar.page_link("pages/2_Patient_Insights.py", label="Patient Insights", icon="💬")
st.sidebar.page_link("pages/3_Demand_Forecasting.py", label="Demand Forecasting", icon="📈")
st.sidebar.page_link("pages/4_Decision_Intelligence.py", label="Decision Panel", icon="🧠")

st.sidebar.subheader("🧪 Clinical Trials")
st.sidebar.page_link("pages/5_Predict_Trial.py", label="Predict Trial", icon="🔮")
st.sidebar.page_link("pages/6_Demo_Trials.py", label="Demo Trials", icon="📂")
st.sidebar.page_link("pages/7_About_Model.py", label="About Model", icon="ℹ️")

# Main Page
st.title("📈 Demand Prediction & Trends")
st.markdown("---")

df = load_data()
drug_list = sorted(df['drugName'].unique())
selected_drug = st.selectbox("Select Drug for Forecasting", drug_list)

if selected_drug:
    row = df[df['drugName'] == selected_drug].iloc[0]
    
    # SECTION 1: ML Prediction
    st.header("1. Machine Learning Demand Prediction (XGBoost)")
    ml_model, ml_features = load_ml_models()
    
    if ml_model is not None:
        ml_pred = predict_demand_ml(row, ml_model, ml_features)
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Current Demand Index", f"{row['demand']:.2f}")
        with c2:
            st.metric("ML Predicted Demand", f"{ml_pred:.2f}", delta=f"{ml_pred - row['demand']:.2f}")
    else:
        st.warning("Ensure the following files exist: demand_model.pkl, demand_features.pkl")
    
    st.markdown("---")
    
    # SECTION 2: DL Forecast
    st.header("2. Deep Learning Trend Forecast (LSTM)")
    dl_model, dl_scaler = load_dl_models()
    
    if dl_model is not None:
        historical, forecast = run_lstm_forecast(row['demand'], dl_model, dl_scaler)
        
        # Plot Trend
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=list(range(-9, 1)), 
            y=historical, 
            mode='lines+markers', 
            name='Historical Trend (Last 10)',
            line=dict(color='#007bff', width=3)
        ))
        
        # Forecasted data
        fig.add_trace(go.Scatter(
            x=list(range(1, 6)), 
            y=forecast, 
            mode='lines+markers', 
            name='LSTM Forecast (Next 5)',
            line=dict(color='#28a745', dash='dash', width=3)
        ))
        
        fig.update_layout(
            title=f"Demand Trend Forecasting for {selected_drug}",
            xaxis_title="Time Steps",
            yaxis_title="Demand Score",
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        trend_msg = "positive" if forecast[-1] > historical[-1] else "negative"
        st.success(f"Forecasting complete. The model predicts a {trend_msg} trend for {selected_drug}.")
    else:
        st.warning("Ensure the following files exist: lstm_demand_model.h5, lstm_scaler.pkl")

st.markdown("---")
st.markdown("Built as part of Pharma Decision Intelligence Platform")
