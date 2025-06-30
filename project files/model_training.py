import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# 1️⃣ Load dataset
df = pd.read_csv('dataset/traffic_volume.csv')
print(df.head())
print(df.info())

# 2️⃣ Check & fill missing values
print(df.isnull().sum())

# Fix column names if needed
df.columns = [col.strip().lower() for col in df.columns]

# Fill numeric nulls
if 'temp' in df.columns:
    df['temp'] = df['temp'].fillna(df['temp'].mean())

if 'rain' in df.columns:
    df['rain'] = df['rain'].fillna(df['rain'].mean())

if 'snow' in df.columns:
    df['snow'] = df['snow'].fillna(df['snow'].mean())

# Fill categorical nulls
if 'weather' in df.columns:
    df['weather'] = df['weather'].fillna(df['weather'].mode()[0])

# ✅ FIX date parsing
df['date'] = pd.to_datetime(df['date'], dayfirst=True)  # <-- KEY FIX

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute

df = df.drop(['date', 'time'], axis=1)

# 4️⃣ Encode categoricals
if 'weather' in df.columns or 'holiday' in df.columns:
    df = pd.get_dummies(df, columns=['weather', 'holiday'], drop_first=True)

# 5️⃣ Features & label
X = df.drop('traffic_volume', axis=1)
y = df['traffic_volume']

# 6️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8️⃣ Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9️⃣ Evaluate
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# 🔟 Save model & scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print(X.columns.tolist())
print("✅ Model & scaler saved.")
