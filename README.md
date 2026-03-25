# 🧠 AI Pharma Decision Intelligence Platform

An end-to-end AI-powered system that transforms raw pharmaceutical data into actionable business insights using NLP, Machine Learning, and Deep Learning.

---

## 🚀 Problem Statement

Pharmaceutical companies deal with massive, fragmented data:

- Patient reviews (unstructured text)
- Drug performance metrics (structured data)
- Demand trends and forecasting

However:

- ❌ No unified system exists to convert data → insights → decisions
- ❌ Decision-making is slow and reactive
- ❌ Early risk detection is missing

---

## 🎯 Solution

This project builds a **Pharma Decision Intelligence Platform** that:

- Extracts insights from patient reviews
- Predicts drug effectiveness and demand
- Forecasts future trends using Deep Learning
- Generates business recommendations (Decision Layer)

---

## 🧩 System Architecture

### 1️⃣ NLP Intelligence Layer
- Sentiment Analysis
- Review Aggregation
- Side Effect Extraction

📌 Output:
- Sentiment score
- Top side effects
- Drug-level insights

---

### 2️⃣ Machine Learning Layer

#### ✅ Drug Effectiveness Prediction
- Model: XGBoost
- Output: Effectiveness Score

#### ✅ Demand Prediction
- Model: XGBoost
- Output: Predicted Demand

---

### 3️⃣ Deep Learning Layer

#### ✅ Demand Forecasting
- Model: LSTM
- Captures:
  - Trends
  - Seasonality
  - Time dependencies

---

### 4️⃣ Decision Intelligence Layer 💀

Business logic engine that converts predictions into actions:

Example:

- 📉 Low sentiment + High side effects  
  → "Investigate drug quality"

- 📈 High demand  
  → "Increase supply"

---

## 📊 Key Features

- 📌 Drug Performance Dashboard
- 💬 Patient Sentiment Insights
- 📈 Demand Forecasting (ML + DL)
- 🧠 Decision Recommendation Engine
- 🧪 Clinical Trial Success Prediction

---

## 🖥️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **TensorFlow / Keras (LSTM)**
- **Streamlit (Frontend UI)**

---

## 📁 Project Structure


AI_PHARMA_DECISION_PLATFORM/
│
├── app/
│ ├── app.py
│ └── pages/
│ ├── Drug_Overview.py
│ ├── Patient_Insights.py
│ ├── Demand_Forecasting.py
│ ├── Decision_Intelligence.py
│ ├── Predict_Trial.py
│ ├── Demo_Trials.py
│ └── About_Model.py
│
├── Drug_module/
│ ├── data/
│ │ ├── Final_data/
│ │ ├── clean_drug_reviews.csv
│ │ ├── bert_enriched_reviews.csv
│ │ ├── drug_nlp_summary.csv
│ │ ├── drug_review_train.csv
│ │ └── drug_review_test.csv
│ │
│ ├── Models/
│ │ ├── demand_model.pkl
│ │ ├── effectiveness_model.pkl
│ │ ├── demand_features.pkl
│ │ ├── effectiveness_features.pkl
│ │ ├── lstm_demand_model.h5
│ │ └── lstm_scaler.pkl
│ │
│ └── notebooks/
│ ├── 01_data_collection.ipynb
│ ├── 02_nlp_processing.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_modelling.ipynb
│ ├── 05_explainibility.ipynb
│ ├── 06_decision_engine.ipynb
│ ├── 07_query_engine.ipynb
│ ├── 08_bert_sentiment.ipynb
│ └── 09_lstm_demand_forecasting.ipynb
│
├── Clinical_Trial_module/
│ ├── Data/
│ │ ├── Clinical_processed/
│ │ └── Clinical_Raw/
│ │
│ ├── Models/clinical/
│ │ ├── model.json
│ │ └── columns.pkl
│ │
│ └── Notebooks/
│ ├── 01_problem.ipynb
│ ├── 02_data_cleaning.ipynb
│ ├── 03_eda.ipynb
│ ├── 04_feature_engineering.ipynb
│ ├── 05_model.ipynb
│ ├── 06_evaluation.ipynb
│ └── 07_explainibility.ipynb
│
├── requirements.txt
├── RUN_PROJECT.txt
└── README.md


---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
streamlit run app/app.py
