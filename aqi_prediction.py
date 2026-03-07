# -*- coding: utf-8 -*-
"""
AQI Prediction Project
An end-to-end Machine Learning project to predict Air Quality Index (AQI) 
using pollutant features.
"""

# ==========================================
# 1. Data Loading & Setup
# ==========================================
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Attempt to download via Kaggle API if not present locally
if not os.path.exists("city_day.csv") and not os.path.exists("/content/city_day.csv"):
    print("Dataset not found locally. Attempting to download via Kaggle API...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_file('rohanrao/air-quality-data-in-india', 'city_day.csv', path='./')
        
        # Check if downloaded as zip and extract
        if os.path.exists("city_day.csv.zip"):
            with zipfile.ZipFile("city_day.csv.zip", 'r') as zip_ref:
                zip_ref.extractall("./")
            os.remove("city_day.csv.zip")
        print("Successfully downloaded city_day.csv using Kaggle API.")
    except Exception as e:
        print(f"Warning: Kaggle API download failed ({e}).")
        print("Please ensure 'city_day.csv' is downloaded from https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india and placed in the current directory.")

# This logic supports running in Google Colab or local Jupyter Notebooks
if os.path.exists("/content/city_day.csv"):
    data = pd.read_csv("/content/city_day.csv")
elif os.path.exists("city_day.csv"):
    data = pd.read_csv("city_day.csv")
else:
    # Use a generic fallback if both don't exist, though it will error if the file is truly absent
    data = pd.read_csv("city_day.csv")

print("First 5 rows:")
print(data.head())

print("\nColumns in the dataset:")
print(data.columns)

# ==========================================
# 2. Data Cleaning
# ==========================================
# Target variable: AQI
# Feature variables: pollutants
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
target = 'AQI'

# Keep only our selected columns
df = data[features + [target]].copy()

# Drop rows where target (AQI) is missing
df = df.dropna(subset=[target])

# For simplicity, fill remaining missing values in features with the column mean
df[features] = df[features].fillna(df[features].mean())

print(f"\nData shape after cleaning: {df.shape}")

# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================
print("\n--- Generating EDA visualizations ---")

# 3.1 AQI distribution histogram
plt.figure(figsize=(8, 5))
sns.histplot(df[target], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Air Quality Index (AQI)")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()

# 3.2 Correlation heatmap between pollutants and AQI
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Pollutants vs AQI")
plt.show()

# 3.3 Scatter plots showing relationships between pollutants and AQI
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.scatterplot(x=df[feature], y=df[target], ax=axes[i], alpha=0.5, color='coral')
    axes[i].set_title(f"{feature} vs AQI")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("AQI")

plt.tight_layout()
plt.show()

# ==========================================
# 4. Feature Selection & Data Splitting
# ==========================================
X = df[features]
y = df[target]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==========================================
# 5. Model Training & Comparison
# ==========================================
# Initialize the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Dictionary to store the results
results_dict = []
trained_models = {}

print("\n--- Training Models ---")
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results_dict.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# ==========================================
# 6. Model Evaluation Results
# ==========================================
results_df = pd.DataFrame(results_dict)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Select the best model (Random Forest is usually best here)
best_model_name = "Random Forest"
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)

# ==========================================
# 7. Visualization: Feature Importance & Actual vs Predicted
# ==========================================
print("\n--- Generating Model Visualizations ---")

# 7.1 Feature Importance (using Random Forest)
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
plt.title("Feature Importance in Predicting AQI (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Pollutant")
plt.show()

# 7.2 Actual vs Predicted Visualization (using Best Model)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, alpha=0.5, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Diagonal line for perfect prediction
plt.title(f"Actual vs Predicted AQI ({best_model_name})")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.show()

# ==========================================
# 8. AQI Prediction Function & Interpretation
# ==========================================
def get_aqi_category(aqi_value):
    """
    Converts a predicted AQI value into a standard category.
    Includes simple health recommendations.
    """
    if aqi_value <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi_value <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."

def predict_aqi(pm25, pm10, no2, so2, o3, co):
    """
    Predicts the AQI based on inputted pollutant levels using the trained Random Forest model.
    """
    # Create DataFrame for single prediction matching the feature order
    input_data = pd.DataFrame([[pm25, pm10, no2, so2, o3, co]], columns=features)
    predicted_value = best_model.predict(input_data)[0]
    category, recommendation = get_aqi_category(predicted_value)
    
    return predicted_value, category, recommendation

# Example Usage
print("\n--- Example AQI Prediction ---")
sample_pm25 = 45.0
sample_pm10 = 120.0
sample_no2 = 30.0
sample_so2 = 15.0
sample_o3 = 40.0
sample_co = 1.2

pred_aqi, cat, rec = predict_aqi(sample_pm25, sample_pm10, sample_no2, sample_so2, sample_o3, sample_co)
print(f"Input Pollutants - PM2.5: {sample_pm25}, PM10: {sample_pm10}, NO2: {sample_no2}, SO2: {sample_so2}, O3: {sample_o3}, CO: {sample_co}")
print(f"Predicted AQI: {pred_aqi:.2f}")
print(f"AQI Category: {cat}")
print(f"Recommendation: {rec}")
