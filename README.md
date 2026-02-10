# 📈 Inflation Forecasting with Financial News Sentiment

End-to-end machine learning pipeline that forecasts U.S. inflation by combining **macroeconomic CPI data** with **transformer-based financial news sentiment (FinBERT)**.  
Includes walk-forward backtesting, SHAP explainability, and an interactive Streamlit dashboard.

---

## 🔎 Project Overview

This project models inflation dynamics using both:

- **Macroeconomic data:** U.S. CPI from FRED
- **NLP signals:** Financial news sentiment extracted using FinBERT
- **Model:** XGBoost regression
- **Validation:** Walk-forward time-series backtesting
- **Explainability:** SHAP feature attribution
- **Interface:** Interactive Streamlit dashboard

---

## 🚀 Live Demo

👉 (Will be added after deployment)

---

## 📊 Key Features

- End-to-end ML pipeline (data → features → model → dashboard)
- Real CPI inflation data from FRED
- Transformer-based financial sentiment integration
- Walk-forward backtesting (time-series safe validation)
- SHAP explainability (global + directional)
- Future inflation forecast
- Interactive dashboard with downloadable predictions

---

## 📉 Model Performance

- **Walk-forward MAE:** ~0.012
- Inflation persistence (lagged inflation) is the strongest predictor
- News sentiment contributes directional influence on inflation forecasts

---

## 🧠 Methodology

### Data
- CPI (FRED): monthly YoY inflation
- Financial news sentiment: FinBERT transformer model

### Features
- Sentiment (current + lagged)
- Lagged inflation values
- Multi-lag time-series structure

### Model
- XGBoost Regressor
- Walk-forward validation
- SHAP for interpretability

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
# 📈 Inflation Forecasting with Financial News Sentiment

End-to-end machine learning pipeline that forecasts U.S. inflation by combining **macroeconomic CPI data** with **transformer-based financial news sentiment (FinBERT)**.  
Includes walk-forward backtesting, SHAP explainability, and an interactive Streamlit dashboard.

---

## 🔎 Project Overview

This project models inflation dynamics using both:

- **Macroeconomic data:** U.S. CPI from FRED
- **NLP signals:** Financial news sentiment extracted using FinBERT
- **Model:** XGBoost regression
- **Validation:** Walk-forward time-series backtesting
- **Explainability:** SHAP feature attribution
- **Interface:** Interactive Streamlit dashboard

---

## 🚀 Live Demo

👉 (Will be added after deployment)

---

## 📊 Key Features

- End-to-end ML pipeline (data → features → model → dashboard)
- Real CPI inflation data from FRED
- Transformer-based financial sentiment integration
- Walk-forward backtesting (time-series safe validation)
- SHAP explainability (global + directional)
- Future inflation forecast
- Interactive dashboard with downloadable predictions

---

## 📉 Model Performance

- **Walk-forward MAE:** ~0.012
- Inflation persistence (lagged inflation) is the strongest predictor
- News sentiment contributes directional influence on inflation forecasts

---

## 🧠 Methodology

### Data
- CPI (FRED): monthly YoY inflation
- Financial news sentiment: FinBERT transformer model

### Features
- Sentiment (current + lagged)
- Lagged inflation values
- Multi-lag time-series structure

### Model
- XGBoost Regressor
- Walk-forward validation
- SHAP for interpretability

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
