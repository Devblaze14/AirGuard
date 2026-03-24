import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading dataset...")
# Load dataset
if os.path.exists("city_day.csv"):
    data = pd.read_csv("city_day.csv")
else:
    print("Please download city_day.csv first!")
    exit(1)

# Clean and prepare
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
target = 'AQI'
df = data[features + [target]].copy().dropna(subset=[target])
df[features] = df[features].fillna(df[features].mean())

X = df[features]
y = df[target]

print("Training lightweight Random Forest model...")
model = RandomForestRegressor(n_estimators=15, max_depth=10, random_state=42)
model.fit(X, y)

# Create an api/ directory if it doesn't exist to save our model
os.makedirs("api", exist_ok=True)

# Save the model
model_path = os.path.join("api", "aqi_model.joblib")
joblib.dump(model, model_path)
print(f"Model successfully saved to {model_path}!")

