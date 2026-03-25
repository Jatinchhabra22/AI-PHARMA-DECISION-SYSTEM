import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="AI Pharma Decision Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND (PURPLE DARK GRADIENT) ===== */
.stApp {
    background: #020617;
    color: #f8fafc;
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Background mesh gradient */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: 
        radial-gradient(circle at 0% 0%, rgba(124, 58, 237, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 100% 100%, rgba(219, 39, 119, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(30, 64, 175, 0.1) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
}

/* ===== HERO SECTION ===== */
.hero-container {
    padding: 100px 20px 60px 20px;
    text-align: center;
    max-width: 1200px;
    margin: 0 auto;
}

.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 100px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 24px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 7rem;
    font-weight: 900;
    line-height: 0.9;
    letter-spacing: -0.05em;
    margin-bottom: 24px;
    background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
}

.hero-subtitle {
    font-size: 1.5rem;
    color: #94a3b8;
    max-width: 700px;
    margin: 0 auto 48px auto;
    line-height: 1.6;
    font-weight: 400;
}

/* ===== GRID CARDS ===== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 24px;
    padding: 40px 0;
    max-width: 1200px;
    margin: 0 auto;
}

.feature-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 32px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
    position: relative;
    overflow: hidden;
}

.feature-card:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(255, 255, 255, 0.1);
    transform: translateY(-8px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
}

.feature-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: radial-gradient(circle at top right, rgba(168, 85, 247, 0.1), transparent 70%);
    opacity: 0;
    transition: opacity 0.4s ease;
}

.feature-card:hover::after {
    opacity: 1;
}

.card-icon {
    font-size: 2rem;
    margin-bottom: 20px;
    display: block;
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 12px;
}

.card-desc {
    font-size: 0.95rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: #ffffff !important;
    color: #020617 !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.2s ease !important;
    font-size: 1rem !important;
}

.stButton > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.2) !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

</style>
""", unsafe_allow_html=True)


def main():
    # ===== SIDEBAR NAV =====
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

    # ===== HERO =====
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">AI-POWERED PHARMA ANALYTICS v2.0</div>
        <div class="hero-title">Precision Medicine.<br>Accelerated.</div>
        <div class="hero-subtitle">
            A unified intelligence platform transforming complex clinical data into 
            actionable strategic decisions using state-of-the-art predictive modeling.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== FEATURE GRID =====
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <span class="card-icon">🧠</span>
            <div class="card-title">Decision Engine</div>
            <div class="card-desc">Advanced heuristic layers converting raw data into executive business recommendations.</div>
        </div>
        <div class="feature-card">
            <span class="card-icon">💬</span>
            <div class="card-title">Patient Voice</div>
            <div class="card-desc">NLP-driven sentiment analysis extracted from thousands of real-world patient reviews.</div>
        </div>
        <div class="feature-card">
            <span class="card-icon">🔮</span>
            <div class="card-title">Trial Prediction</div>
            <div class="card-desc">Machine learning models estimating clinical trial success probabilities with high precision.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ===== ACTION BUTTONS =====
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Get Started", use_container_width=True):
            st.switch_page("pages/1_Drug_Intelligence.py")

    # ===== FOOTER =====
    st.markdown("""
    <div style="text-align: center; margin-top: 100px; padding: 40px; color: #475569; font-size: 0.9rem; border-top: 1px solid rgba(255,255,255,0.05);">
        © 2026 AI Pharma Decision Platform. Built for Precision & Scale.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#94a3b8;'>Built as part of the Pharma Decision Intelligence Platform | © 2026</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()