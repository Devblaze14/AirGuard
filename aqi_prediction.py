# -*- coding: utf-8 -*-
"""
AirGuard – Air Quality Prediction and Health Advisory System
An end-to-end Machine Learning project to predict Air Quality Index (AQI) 
using pollutant features and provide actionable health recommendations.
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

# Create an outputs directory for saving graphs
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Dataset Loading ---
# The logic below attempts to automatically load or download the dataset using the Kaggle API.
# If the file 'city_day.csv' is not found locally, it connects to Kaggle to fetch it.
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

# Read the dataset into a pandas DataFrame
# This logic supports running in Google Colab (/content/) or local environments
if os.path.exists("/content/city_day.csv"):
    data = pd.read_csv("/content/city_day.csv")
elif os.path.exists("city_day.csv"):
    data = pd.read_csv("city_day.csv")
else:
    # Fallback to local
    data = pd.read_csv("city_day.csv")

print("First 5 rows:")
print(data.head())

# ==========================================
# 2. Data Cleaning
# ==========================================
# Target variable: The feature we want to predict
target = 'AQI'

# Feature variables: The inputs used to make the prediction
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']

# Keep only our selected columns
df = data[features + [target]].copy()

# --- Handle Missing Values ---
# 1. Drop rows where the target (AQI) is missing, as we cannot train without a target label
df = df.dropna(subset=[target])

# 2. Fill remaining missing values in the feature columns with the mean of each respective column
df[features] = df[features].fillna(df[features].mean())

print(f"\nData shape after cleaning: {df.shape}")

# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================
# EDA helps us understand the underlying patterns and relationships in our data.
print("\n--- Generating EDA visualizations ---")

# 3.1 AQI distribution histogram
# This shows the frequency of different AQI values across our dataset.
plt.figure(figsize=(8, 5))
sns.histplot(df[target], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Air Quality Index (AQI)")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, 'aqi_distribution.png'), bbox_inches='tight')
plt.show()

# 3.2 Correlation heatmap between pollutants and AQI
# This heatmap reveals which pollutants have the strongest linear relationship with AQI.
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Pollutants vs AQI")
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight')
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
plt.savefig(os.path.join(output_dir, 'pollutant_scatter_plots.png'), bbox_inches='tight')
plt.show()

# ==========================================
# 4. Feature Selection & Data Splitting
# ==========================================
X = df[features]
y = df[target]

# Train-test split (80% train, 20% test)
# This allows us to train the model on one portion of the data and evaluate its performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==========================================
# 5. Model Training & Comparison
# ==========================================
# We initialize three different machine learning models to compare their performance.
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Dictionary to store the evaluation results
results_dict = []
trained_models = {}

print("\n--- Training Models ---")
for name, model in models.items():
    # Train the model on the training data
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Predict AQI values for the testing data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    # MAE (Mean Absolute Error): Average absolute difference between predicted and actual values.
    # RMSE (Root Mean Squared Error): Standard deviation of the prediction errors.
    # R2 Score: Proportion of the variance in the dependent variable that is predictable.
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
# Display the performance comparison of all models
results_df = pd.DataFrame(results_dict)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Select the best model (Random Forest typically performs the best for this dataset)
best_model_name = "Random Forest"
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)

# ==========================================
# 7. Visualization: Feature Importance & Actual vs Predicted
# ==========================================
print("\n--- Generating Model Visualizations ---")

# 7.1 Feature Importance (using Random Forest)
# Shows which pollutants were most useful to the model in predicting AQI.
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
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
plt.show()

# 7.2 Actual vs Predicted Visualization (using Best Model)
# Visually compares the model's predictions against the real test-set values.
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, alpha=0.5, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Diagonal line for perfect prediction
plt.title(f"Actual vs Predicted AQI ({best_model_name})")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), bbox_inches='tight')
plt.show()

# ==========================================
# 8. AQI Prediction Function & Interpretation
# ==========================================
def get_aqi_category(aqi_value):
    """
    Converts a predicted AQI value into a standard category.
    Provides simple, actionable health recommendations based on air quality.
    """
    if aqi_value <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "Sensitive groups should reduce outdoor activity. The general public is less likely to be affected."
    elif aqi_value <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."

def predict_aqi(pm25, pm10, no2, so2, o3, co):
    """
    Predicts the AQI based on user-inputted pollutant levels using the trained Random Forest model.
    Returns the numeric prediction, categorical classification, and a health advisory.
    """
    # Create DataFrame for single prediction matching the feature order
    input_data = pd.DataFrame([[pm25, pm10, no2, so2, o3, co]], columns=features)
    predicted_value = best_model.predict(input_data)[0]
    category, recommendation = get_aqi_category(predicted_value)
    
    return predicted_value, category, recommendation

# Example Usage
print("\n--- Example Prediction Output ---")
sample_pm25 = 45.0
sample_pm10 = 120.0
sample_no2 = 30.0
sample_so2 = 15.0
sample_o3 = 40.0
sample_co = 1.2

pred_aqi, cat, rec = predict_aqi(sample_pm25, sample_pm10, sample_no2, sample_so2, sample_o3, sample_co)
print(f"Predicted AQI: {int(round(pred_aqi))}")
print(f"Category: {cat}")
print(f"Recommendation: {rec}")
