import streamlit as st
import numpy as np
import joblib

# Load saved scaler and models
scaler = joblib.load("saved_models/scaler.pkl")
models = {
    "Logistic Regression": joblib.load("saved_models/logistic_regression.pkl"),
    "Random Forest": joblib.load("saved_models/random_forest.pkl"),
    "XGBoost": joblib.load("saved_models/xgboost.pkl")
}

st.set_page_config(page_title="📈 Stock Price Prediction App", layout="centered")

st.title("📈 Stock Price Prediction App")
st.write("Enter stock features below and choose a model to predict the **target (price direction or class)**.")

# Define input fields for all features
st.subheader("🔢 Input Features")

open_ = st.number_input("Open Price", value=0.0)
high = st.number_input("High Price", value=0.0)
low = st.number_input("Low Price", value=0.0)
close = st.number_input("Close Price", value=0.0)
volume = st.number_input("Volume", value=0.0)
daily_return = st.number_input("Daily Return", value=0.0)
cumulative_return = st.number_input("Cumulative Return", value=0.0)
price_range = st.number_input("Price Range", value=0.0)
pct_price_range = st.number_input("Percent Price Range", value=0.0)
ma5 = st.number_input("5-Day Moving Average", value=0.0)
ma20 = st.number_input("20-Day Moving Average", value=0.0)
ma50 = st.number_input("50-Day Moving Average", value=0.0)
volatility5 = st.number_input("5-Day Volatility", value=0.0)
volatility20 = st.number_input("20-Day Volatility", value=0.0)
rsi = st.number_input("RSI (Relative Strength Index)", value=0.0)

# Select model
model_name = st.selectbox("Select Model", list(models.keys()))

# Prediction button
if st.button("Predict"):
    features = np.array([[
        open_, high, low, close, volume,
        daily_return, cumulative_return, price_range,
        pct_price_range, ma5, ma20, ma50,
        volatility5, volatility20, rsi
    ]])

    # Apply scaler only for Logistic Regression
    if model_name == "Logistic Regression":
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    model = models[model_name]
    prediction = model.predict(features_scaled)

    st.success(f"✅ Model: {model_name}")
    st.success(f"📊 Predicted Target: {prediction[0]}")
    st.balloons()

st.markdown("---")
st.caption("Developed by **Marwan Eslam Ouda**")
st.caption("📊 Data Source: [Kaggle - Stock Market Performance Dashboard Comparing Tech Giants](https://www.kaggle.com/datasets/mirichoi0218/stock-market-data)")
