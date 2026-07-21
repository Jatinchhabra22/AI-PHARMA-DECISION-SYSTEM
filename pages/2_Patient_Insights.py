import streamlit as st
import pandas as pd
import plotly.express as px
import os
import ast

# Page config
st.set_page_config(page_title="Patient Insights", page_icon="💬", layout="wide")

# Paths
DATA_PATH = os.path.join("Drug_module", "data", "Final_data", "drug_final_enriched_dataset.csv")

# Functions
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

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
st.title("🗣️ Patient Insights & Sentiment Analysis")
st.markdown("---")

df = load_data()
drug_list = sorted(df['drugName'].unique())
selected_drug = st.selectbox("Select Drug for Detailed Insights", drug_list)

# FIXED mapping for colors
color_map = {
    "positive": "green",
    "neutral": "blue",
    "negative": "red"
}

if selected_drug:
    row = df[df['drugName'] == selected_drug].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Analysis")
        sentiment_score = row['avg_sentiment']
        
        # Fixed logic for sentiment breakdown
        if sentiment_score > 0.05:
            label = "positive"
        elif sentiment_score < -0.05:
            label = "negative"
        else:
            label = "neutral"
            
        fig = px.pie(
            names=[label.capitalize(), "Other"],
            values=[abs(sentiment_score) if abs(sentiment_score) > 0 else 0.1, 1 - abs(sentiment_score)],
            color=[label.capitalize(), "Other"],
            color_discrete_map={
                label.capitalize(): color_map[label],
                "Other": "#e9ecef"
            },
            hole=0.4
        )
        fig.update_layout(title_text=f"Average Sentiment: {sentiment_score:.2f} ({label.capitalize()})")
        st.plotly_chart(fig)
        
    with col2:
        st.subheader("Reported Side Effects")
        st.info(format_side_effects(row['top_side_effects']))
        
        st.subheader("Key Patient Insights")
        insights = ast.literal_eval(row['insights']) if isinstance(row['insights'], str) else row['insights']
        if insights:
            for ins in insights:
                st.warning(ins)
        else:
            st.write("No specific insights available.")
            
    st.markdown("---")
    
    # Sentiment over review count or engagement
    st.subheader("Patient Engagement vs Sentiment")
    fig2 = px.scatter(
        df, 
        x="engagement_score", 
        y="avg_sentiment", 
        color="drug_category",
        hover_data=['drugName'],
        title="Engagement vs Sentiment for All Drugs"
    )
    # Highlight selected drug
    fig2.add_scatter(
        x=[row['engagement_score']], 
        y=[row['avg_sentiment']], 
        mode='markers', 
        marker=dict(size=15, color='black', symbol='star'),
        name=f"Selected: {selected_drug}"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("Built as part of Pharma Decision Intelligence Platform")
