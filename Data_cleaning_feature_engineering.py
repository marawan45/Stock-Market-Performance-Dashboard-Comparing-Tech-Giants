import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("C://Users//maraw//Downloads//Stock Market Performance Dashboard  Comparing Tech Giants//all_stocks_5yr.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip().str.lower()

# Fix numeric columns by removing $, commas, and spaces
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(str).replace({'\$': '', ',': '', ' ': ''}, regex=True)

# Convert to proper numeric types
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
df['volume'] = df['volume'].astype(int)

# Parse date correctly with dayfirst=True
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Drop invalid dates and missing rows
df = df.dropna(subset=['date'])
df.ffill(inplace=True)
df.drop_duplicates(inplace=True)

# Feature engineering
df['Daily_Return'] = df['close'].pct_change()
df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
df['Price_Range'] = df['high'] - df['low']
df['Pct_Price_Range'] = (df['Price_Range'] / df['low']) * 100

# Moving averages
df['MA_5'] = df['close'].rolling(window=5).mean()
df['MA_20'] = df['close'].rolling(window=20).mean()
df['MA_50'] = df['close'].rolling(window=50).mean()

# Volatility
df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

# RSI Calculation
delta = df['close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Binary target for ML
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Extract date parts
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day
df['Weekday'] = df['date'].dt.day_name()
df['Month_Name'] = df['date'].dt.month_name()

# Drop rows with missing values caused by rolling windows
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("clean_sp500_final.csv", index=False)
print(" Cleaned and enriched dataset saved as clean_sp500_final.csv")
