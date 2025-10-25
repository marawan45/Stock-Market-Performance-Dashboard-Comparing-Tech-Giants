# 📈 Stock Market Performance Dashboard - Comparing Tech Giants (S&P 500)

A comprehensive end-to-end data science project that analyzes S&P 500 stock market data, featuring data cleaning, exploratory analysis, interactive visualizations, and machine learning models for stock price prediction.

## Dash
<img width="1227" height="704" alt="image" src="https://github.com/user-attachments/assets/c36685df-49af-44fb-84fd-614a9845f1f3" />




## 🎯 Project Overview

This project provides a complete pipeline for stock market data analysis, from raw data processing to machine learning predictions. It includes interactive dashboards built with Power BI/Tableau and a Streamlit web application for real-time stock price predictions.

## ✨ Features

### 📊 Data Analysis & Visualization
- Interactive dashboard comparing tech giants' performance
- Cumulative returns, daily returns, and volume analysis
- RSI (Relative Strength Index) indicators
- Time-series trends and weekday performance patterns
- Multi-stock comparison across years

### 🤖 Machine Learning Models
- **Logistic Regression**: Binary classification for price direction
- **Random Forest**: Ensemble learning with feature importance analysis
- **XGBoost**: Gradient boosting for improved accuracy
- Real-time prediction via Streamlit web application

### 🧹 Data Processing
- Automated data cleaning and validation
- Feature engineering (Moving Averages, RSI, Volatility)
- SQL-based exploratory data analysis
- Handling of missing values and outliers

## 🛠️ Tech Stack

**Languages & Libraries:**
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib, Seaborn
- SQL

**Tools:**
- Power BI / Tableau (Dashboard)
- Jupyter Notebook
- Git & GitHub

## 📁 Project Structure

```
├── Data_cleaning_feature_engineering.py    # Data preprocessing and feature creation
├── Data_exploration_cleaning.sql           # SQL queries for data exploration
├── machine_learning_models.py              # ML model training and evaluation
├── app.py                                  # Streamlit web application
├── saved_models/                           # Trained models and scaler
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── clean_sp500_final.csv                   # Processed dataset
└── README.md                               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
pip
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-market-dashboard.git
cd stock-market-dashboard
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn joblib
```



### Usage

#### 1. Data Cleaning and Feature Engineering
```bash
python Data_cleaning_feature_engineering.py
```
This script will:
- Clean column names and fix data types
- Handle missing values and duplicates
- Create technical indicators (MA, RSI, Volatility)
- Generate `clean_sp500_final.csv`

#### 2. Train Machine Learning Models
```bash
python machine_learning_models.py
```
This will:
- Train three classification models
- Display performance metrics and confusion matrices
- Save models to `saved_models/` directory

#### 3. Run the Prediction Web App
```bash
streamlit run app.py
```
Access the app at `http://localhost:8501`

## 📈 Features Engineered

| Feature | Description |
|---------|-------------|
| `Daily_Return` | Percentage change in closing price |
| `Cumulative_Return` | Cumulative product of daily returns |
| `Price_Range` | Difference between high and low prices |
| `MA_5, MA_20, MA_50` | Moving averages (5, 20, 50 days) |
| `Volatility_5, Volatility_20` | Rolling standard deviation of returns |
| `RSI` | Relative Strength Index (14-day period) |
| `Target` | Binary label: 1 if next day's price increases, 0 otherwise |


## 📊 Dashboard Insights

The interactive dashboard provides:
- **Cumulative Returns**: 3.49M across all stocks
- **Total Volume**: 3T shares traded
- **Average RSI**: 52.95 indicating neutral market conditions
- **Weekday Patterns**: Friday and Wednesday show highest average returns
- **Year-over-Year Trends**: Visual comparison of stock performance from 2013-2018

## 🔍 Key Findings

1. Tech giants show strong cumulative returns over the 5-year period
2. Friday and Wednesday demonstrate the highest average daily returns
3. Volume patterns indicate increased trading activity in specific months
4. RSI indicators help identify overbought/oversold conditions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Marwan Eslam Ouda**

## 🙏 Acknowledgments

- Dataset: [Kaggle - Stock Market Data](https://www.kaggle.com/datasets/mirichoi0218/stock-market-data)
- Inspiration: Financial market analysis and algorithmic trading
- Libraries: Scikit-learn, XGBoost, Streamlit, and the Python community



---

⭐ If you found this project helpful, please consider giving it a star!
