import streamlit as st
import pandas as pd
import os
import ast

# Page config
st.set_page_config(page_title="Decision Intelligence", page_icon="🧠", layout="wide")

# Paths
DATA_PATH = os.path.join("Drug_module", "data", "Final_data", "drug_final_enriched_dataset.csv")

# Functions
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

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
st.title("🧠 Decision Intelligence Panel")
st.markdown("---")

df = load_data()
drug_list = sorted(df['drugName'].unique())
selected_drug = st.selectbox("Select Drug for Decision Analysis", drug_list)

if selected_drug:
    try:
        row = df[df['drugName'] == selected_drug].iloc[0]
        
        # Decision Engine Logic
        performance_score = row.get('performance_score', 0.0)
        risk_score = row.get('risk_score', 0.0)
        opportunity_score = row.get('opportunity_score', 0.0)
        
        st.container()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Performance Score", f"{performance_score:.2f}")
        with col2:
            st.metric("Risk Score", f"{risk_score:.2f}")
        with col3:
            st.metric("Opportunity Score", f"{opportunity_score:.2f}")
            
        st.markdown("---")
        
        # Color Coding Decision Box
        # Apply color coding:
        # High Performer → green
        # High Risk → red
        # Opportunity → yellow
        
        if performance_score > 2.5:
            st.success(f"### 📈 {selected_drug} - High Performer")
            st.write("This drug demonstrates strong performance across sentiment, effectiveness, and demand.")
        elif risk_score > 0.5:
            st.error(f"### ⚠️ {selected_drug} - High Risk")
            st.write("Caution advised. This drug shows potential risks in terms of patient sentiment or side effects.")
        elif opportunity_score > 0.5:
            st.warning(f"### 💡 {selected_drug} - Opportunity")
            st.write("This drug represents a strategic opportunity for market growth and development.")
        else:
            st.info(f"### ⚖️ {selected_drug} - Balanced Drug")
            st.write("This drug maintains a stable performance with balanced risk and opportunity metrics.")

        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Key Insights")
            insights = row.get('insights', "[]")
            if isinstance(insights, str):
                try:
                    insights = ast.literal_eval(insights)
                except Exception:
                    insights = [insights]
            
            if not insights or (isinstance(insights, list) and len(insights) == 0):
                st.write("No specific insights recorded.")
            else:
                for ins in insights:
                    st.info(f"• {ins}")
                
        with c2:
            st.subheader("Recommended Actions")
            actions = row.get('actions', "[]")
            if isinstance(actions, str):
                try:
                    actions = ast.literal_eval(actions)
                except Exception:
                    actions = [actions]
            
            if not actions or (isinstance(actions, list) and len(actions) == 0):
                st.write("No specific actions recommended.")
            else:
                for act in actions:
                    st.success(f"• {act}")
                    
    except Exception as e:
        st.error(f"Decision Intelligence Error: {str(e)}")

st.markdown("---")
st.markdown("Built as part of Pharma Decision Intelligence Platform")
