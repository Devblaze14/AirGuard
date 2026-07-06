# -*- coding: utf-8 -*-
"""
AirGuard – Air Quality Prediction and Health Advisory System
An end-to-end Machine Learning project to predict Air Quality Index (AQI) 
using pollutant features and provide actionable health recommendations.
"""

# ==========================================
# 1. Imports & Config
# ==========================================
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Create an outputs directory for saving graphs
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Module-level feature list and target name
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
target = 'AQI'

# Module-level variables populated by init()
data = None          # Raw dataframe (with City, Date columns)
df = None            # Cleaned dataframe (features + target)
X_train = X_test = y_train = y_test = None
best_model = None
best_model_name = "Random Forest"
trained_models = {}
_is_initialized = False


# ==========================================
# 2. Data Loading & Cleaning
# ==========================================
def load_data():
    """Load the city_day.csv dataset, trying multiple paths."""
    if not os.path.exists("city_day.csv") and not os.path.exists("/content/city_day.csv"):
        print("Dataset not found locally. Attempting to download via Kaggle API...")
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_file(
                'rohanrao/air-quality-data-in-india', 'city_day.csv', path='./'
            )
            if os.path.exists("city_day.csv.zip"):
                with zipfile.ZipFile("city_day.csv.zip", 'r') as zip_ref:
                    zip_ref.extractall("./")
                os.remove("city_day.csv.zip")
            print("Successfully downloaded city_day.csv using Kaggle API.")
        except Exception as e:
            print(f"Warning: Kaggle API download failed ({e}).")
            print("Please ensure 'city_day.csv' is downloaded and placed in the current directory.")

    if os.path.exists("/content/city_day.csv"):
        raw = pd.read_csv("/content/city_day.csv")
    else:
        raw = pd.read_csv("city_day.csv")

    print("First 5 rows:")
    print(raw.head())
    return raw


def clean_data(raw_df):
    """Clean the dataset: keep features + target, drop NaN targets, fill feature NaNs."""
    cleaned = raw_df[features + [target]].copy()
    cleaned = cleaned.dropna(subset=[target])
    cleaned[features] = cleaned[features].fillna(cleaned[features].mean())
    print(f"\nData shape after cleaning: {cleaned.shape}")
    return cleaned


# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================
def run_eda(cleaned_df):
    """Generate and save EDA visualizations."""
    print("\n--- Generating EDA visualizations ---")

    # 3.1 AQI distribution histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(cleaned_df[target], bins=50, kde=True, color='skyblue')
    plt.title("Distribution of Air Quality Index (AQI)")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, 'aqi_distribution.png'), bbox_inches='tight')
    plt.close()

    # 3.2 Correlation heatmap between pollutants and AQI
    plt.figure(figsize=(8, 6))
    correlation_matrix = cleaned_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap: Pollutants vs AQI")
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close()

    # 3.3 Scatter plots showing relationships between pollutants and AQI
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.scatterplot(x=cleaned_df[feature], y=cleaned_df[target], ax=axes[i], alpha=0.5, color='coral')
        axes[i].set_title(f"{feature} vs AQI")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pollutant_scatter_plots.png'), bbox_inches='tight')
    plt.close()


# ==========================================
# 4. Model Training & Comparison
# ==========================================
def _compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error, ignoring zeros."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_models(X_tr, y_tr, X_te, y_te):
    """Train all models (including baseline), return results list and trained models dict."""
    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results_list = []
    models_trained = {}

    print("\n--- Training Models ---")
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        models_trained[name] = model

        y_pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        mape = _compute_mape(y_te.values, y_pred)

        results_list.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R² Score": round(r2, 4),
            "MAPE (%)": round(mape, 2),
        })

    results_df = pd.DataFrame(results_list)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    return results_df, models_trained


# ==========================================
# 5. Residual Analysis
# ==========================================
def residual_analysis(model, X_te, y_te, raw_df=None):
    """
    Save a residual plot and print error broken down by AQI category.
    Optionally breaks down by city if raw_df (with 'City' column) is available.
    """
    y_pred = model.predict(X_te)
    residuals = y_te.values - y_pred

    # --- Residual scatter plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.4, color='steelblue', s=15)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.title("Residual Plot (Random Forest)")
    plt.xlabel("Predicted AQI")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), bbox_inches='tight')
    plt.close()
    print(f"\nResidual plot saved to {output_dir}/residual_plot.png")

    # --- Error by AQI category ---
    temp = pd.DataFrame({'actual': y_te.values, 'predicted': y_pred})
    temp['category'] = temp['actual'].apply(lambda v: get_aqi_category(v)[0])
    temp['abs_error'] = np.abs(temp['actual'] - temp['predicted'])

    print("\nMean Absolute Error by AQI Category:")
    cat_errors = temp.groupby('category')['abs_error'].mean().sort_values(ascending=False)
    for cat, err in cat_errors.items():
        print(f"  {cat:35s}  MAE = {err:.2f}")

    # --- Error by city (if city info available) ---
    if raw_df is not None and 'City' in raw_df.columns:
        # Align city labels with the test indices
        city_series = raw_df.loc[y_te.index, 'City'] if y_te.index.isin(raw_df.index).all() else None
        if city_series is not None:
            temp['city'] = city_series.values
            print("\nMean Absolute Error by City (top 10):")
            city_errors = temp.groupby('city')['abs_error'].mean().sort_values(ascending=False).head(10)
            for city, err in city_errors.items():
                print(f"  {city:25s}  MAE = {err:.2f}")


# ==========================================
# 6. SHAP Explainability
# ==========================================
def generate_shap_summary(model, X_te):
    """Compute SHAP values and save a global summary plot to outputs/."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    plt.figure()
    shap.summary_plot(shap_values, X_te, show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {output_dir}/shap_summary.png")
    return explainer


def get_shap_explanation(input_row):
    """
    Return per-feature SHAP contributions for a single prediction.

    Parameters
    ----------
    input_row : dict or pd.DataFrame
        A single row of pollutant values with keys matching `features`.

    Returns
    -------
    dict : {feature_name: signed_contribution} sorted by absolute magnitude (descending).
    """
    import shap
    if isinstance(input_row, dict):
        input_row = pd.DataFrame([input_row], columns=features)

    explainer = shap.TreeExplainer(best_model)
    sv = explainer.shap_values(input_row)[0]  # single row → 1-D array

    contributions = dict(zip(features, sv))
    # Sort by absolute magnitude descending
    contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    return contributions


# ==========================================
# 7. Visualization Helpers
# ==========================================
def plot_feature_importance(model, feat_names):
    """Plot and save feature importance bar chart."""
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(
        by='Importance', ascending=True
    )
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='lightgreen')
    plt.title("Feature Importance in Predicting AQI (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Pollutant")
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Plot and save actual vs predicted scatter."""
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, color='purple')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f"Actual vs Predicted AQI ({model_name})")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), bbox_inches='tight')
    plt.close()


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
    init()  # Ensure model is loaded
    input_data = pd.DataFrame([[pm25, pm10, no2, so2, o3, co]], columns=features)
    predicted_value = best_model.predict(input_data)[0]
    category, recommendation = get_aqi_category(predicted_value)
    return predicted_value, category, recommendation


# ==========================================
# 9. Initialization (lazy, runs once)
# ==========================================
def init():
    """Initialize the module: load data, clean, train models. Runs only once."""
    global data, df, X_train, X_test, y_train, y_test
    global best_model, trained_models, _is_initialized

    if _is_initialized:
        return

    data = load_data()
    df = clean_data(data)

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    _, trained_models = train_models(X_train, y_train, X_test, y_test)
    best_model = trained_models[best_model_name]

    _is_initialized = True


# ==========================================
# 10. Main Entry Point
# ==========================================
if __name__ == "__main__":
    init()

    # EDA
    run_eda(df)

    # Visualizations
    print("\n--- Generating Model Visualizations ---")
    plot_feature_importance(best_model, features)
    best_predictions = best_model.predict(X_test)
    plot_actual_vs_predicted(y_test, best_predictions, best_model_name)

    # Residual analysis
    residual_analysis(best_model, X_test, y_test, raw_df=data)

    # SHAP
    print("\n--- SHAP Explainability ---")
    generate_shap_summary(best_model, X_test)

    # Example prediction
    print("\n--- Example Prediction Output ---")
    sample_pm25, sample_pm10, sample_no2 = 45.0, 120.0, 30.0
    sample_so2, sample_o3, sample_co = 15.0, 40.0, 1.2

    pred_aqi, cat, rec = predict_aqi(sample_pm25, sample_pm10, sample_no2,
                                     sample_so2, sample_o3, sample_co)
    print(f"Predicted AQI: {int(round(pred_aqi))}")
    print(f"Category: {cat}")
    print(f"Recommendation: {rec}")

    # Example SHAP explanation
    sample_row = dict(zip(features, [sample_pm25, sample_pm10, sample_no2,
                                     sample_so2, sample_o3, sample_co]))
    shap_contrib = get_shap_explanation(sample_row)
    print("\nSHAP Contributions (per pollutant):")
    for feat, val in shap_contrib.items():
        print(f"  {feat:10s}: {val:+.2f}")
