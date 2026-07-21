import streamlit as st

# Page config
st.set_page_config(page_title="Technical Documentation", page_icon="ℹ️", layout="wide")

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
st.title("ℹ️ AI Pharma Decision Intelligence Platform")
st.subheader("Comprehensive Technical Documentation & Project Architecture")
st.markdown("---")

# Project Overview
st.header("🌟 Project Overview")
st.write("""
The **AI Pharma Decision Intelligence Platform** is an end-to-end analytical ecosystem designed to revolutionize how pharmaceutical companies evaluate drug performance and clinical trial risks. By unifying **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **Predictive Analytics**, the platform transforms fragmented medical and market data into a cohesive decision-making engine.

Our mission is to reduce the high failure rates in clinical trials and provide deep, data-driven insights into patient experiences across the global drug market.
""")

# Architecture
st.header("🏗️ Core Architecture")
st.write("""
The platform is built on a multi-layered architecture:
1.  **Data Layer**: Ingests raw data from clinical trial registries (ClinicalTrials.gov) and drug review repositories.
2.  **Processing Layer**: Uses NLP pipelines for sentiment extraction, keyword mining, and TF-IDF vectorization.
3.  **Intelligence Layer**: Houses high-performance XGBoost and Deep Learning models for predictive tasks.
4.  **Presentation Layer**: A modern, glassmorphic Streamlit interface designed for executive-level clarity.
""")

# Intelligence Modules
st.header("💊 1. Drug Intelligence & Patient Voice")
st.write("""
This module focuses on the post-market performance of drugs, leveraging real-world evidence.
- **Sentiment Analysis**: Using advanced NLP, we process thousands of patient reviews to derive an `avg_sentiment` score. This reflects the qualitative patient experience beyond clinical data.
- **Effectiveness Scoring**: A heuristic model that combines patient ratings, sentiment, and usefulness metrics to score how well a drug actually works in the real world.
- **Side Effect Mining**: Automated extraction of reported side effects from unstructured text, providing an early warning system for safety profiles.
- **Drug Segmentation**: Categorizes drugs into *High Performers*, *High Risk*, or *Strategic Opportunities* based on a multi-dimensional matrix of demand and sentiment.
""")

st.header("🧪 2. Clinical Trial Prediction Engine")
st.write("""
The "Predict Success" module addresses the R&D bottleneck by estimating the probability of a clinical trial's success before significant capital is committed.
- **Predictive Modeling**: Uses an optimized **XGBoost Classifier** trained on historical trial outcomes.
- **Key Feature Engineering**:
    - **Log Enrollment**: Normalizes trial size to handle the vast variance between Phase 1 and Phase 3 trials.
    - **Temporal Analysis**: Evaluates success probability based on trial duration and phase-specific constraints.
    - **Categorical Encoding**: Handles complex categorical data like funder types (Industry vs. Academic) and study types.
- **Risk Assessment**: Classifies trials into Low, Medium, or High Risk categories with actionable recommendations for R&D managers.
""")

st.header("🧠 3. Decision Intelligence Panel")
st.write("""
The Decision Intelligence layer acts as the "Brain" of the platform, synthesizing outputs from all other modules.
- **Performance vs. Risk Matrix**: Visualizes where a drug or trial stands on the spectrum of market performance and clinical risk.
- **Automated Insights**: Generates strategic recommendations (e.g., "Increase R&D investment" or "Initiate safety review") based on model thresholds.
- **Strategic Opportunity Detection**: Identifies drugs that have high patient sentiment but low market demand, indicating potential for marketing growth.
""")

# Tech Stack
st.header("🛠️ Technology Stack")
st.columns(3)[0].markdown("""
**Frontend & UI**
- Streamlit (v1.32+)
- Custom CSS (Glassmorphism)
- Plotly Express (Interactive Viz)
""")
st.columns(3)[1].markdown("""
**Machine Learning**
- XGBoost (High-performance Boosting)
- Scikit-Learn (Preprocessing)
- Joblib/Pickle (Model Serialization)
""")
st.columns(3)[2].markdown("""
**NLP & Data**
- NLTK/Spacy (Text Processing)
- Pandas/NumPy (Data Engineering)
- TF-IDF Vectorization
""")

# Future Roadmap
st.header("🚀 Future Roadmap")
st.write("""
- **Real-time API Integration**: Directly fetching live data from ClinicalTrials.gov and FDA adverse event reporting systems.
- **Deep Learning Forecasting**: Implementing LSTM (Long Short-Term Memory) networks for multi-year drug demand forecasting.
- **Generative AI Reports**: Using LLMs to generate detailed executive summaries and regulatory filing drafts based on platform insights.
""")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Built for the Future of Pharmaceutical R&D | © 2026 AI Pharma Platform</p>", unsafe_allow_html=True)
