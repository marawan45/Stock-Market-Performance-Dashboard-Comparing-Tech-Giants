import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("C://Users//maraw//Downloads//Stock Market Performance Dashboard  Comparing Tech Giants//clean_sp500_final.csv")

# Select features (independent variables)
features = [
    'open', 'high', 'low', 'close', 'volume',
    'Daily_Return', 'Cumulative_Return', 'Price_Range',
    'Pct_Price_Range', 'MA_5', 'MA_20', 'MA_50',
    'Volatility_5', 'Volatility_20', 'RSI'
]

# Target variable
X = df[features]
y = df['Target']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("hello")
# ------------------ Model 1: Logistic Regression ------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

# ------------------ Model 2: Random Forest ------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ------------------ Model 3: XGBoost ------------------
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# ------------------ Evaluation ------------------
models = {
    "Logistic Regression": log_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred
}

print("\n📊 MODEL PERFORMANCE COMPARISON:\n")
for name, pred in models.items():
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Classification Report:\n", classification_report(y_test, pred))
    print("-" * 50)

# ------------------ Confusion Matrix ------------------
plt.figure(figsize=(15, 4))
for i, (name, pred) in enumerate(models.items()):
    plt.subplot(1, 3, i + 1)
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ------------------ Feature Importance (Random Forest) ------------------
importances = rf_model.feature_importances_
feat_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feat_importance)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

import joblib
import os

# Create a directory to save models if it doesn’t exist
os.makedirs("saved_models", exist_ok=True)

# Save Scaler
joblib.dump(scaler, "saved_models/scaler.pkl")

# Save Models
joblib.dump(log_model, "saved_models/logistic_regression.pkl")
joblib.dump(rf_model, "saved_models/random_forest.pkl")
joblib.dump(xgb_model, "saved_models/xgboost.pkl")

print("✅ All models and scaler have been saved successfully!")
