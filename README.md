# AirGuard – Air Quality Prediction and Health Advisory System

## Project Overview
AirGuard is a Machine Learning project to predict Air Quality Index (AQI) based on pollutant levels. It includes data processing, exploratory data analysis (EDA), model comparison, SHAP explainability, anomaly detection, an AI-powered health advisory, and an interactive Streamlit dashboard.

## Problem Statement
The goal of this system is to accurately map multivariate pollutant inputs to a standardized Air Quality Index (AQI) and provide corresponding health recommendations.

## Features
- **Data Processing**: Automatically loads, cleans, and handles missing values.
- **Exploratory Data Analysis (EDA)**: Generates visual insights like histograms, heatmaps, and scatter plots.
- **Model Training & Comparison**: Tests Baseline (Mean), Linear Regression, Decision Tree, and Random Forest models.
- **Enhanced Evaluation**: Comparison table with MAE, RMSE, R², and MAPE. Residual analysis with per-category and per-city error breakdown.
- **SHAP Explainability**: Global SHAP summary plot and per-prediction feature contributions.
- **Anomaly Detection**: Flags pollution spikes using Isolation Forest.
- **GenAI Health Advisory**: LLM-powered (Groq) health advisory with offline fallback.
- **Streamlit Dashboard**: Interactive panels for AQI trends, predictions, anomalies, and advisories.
- **Model Insights**: Analyzes and plots feature importance for AQI predictions.
- **Automated Kaggle Integration**: Can pull the dataset securely via the Kaggle API.
- **Actionable Prediction**: Offers a reusable `predict_aqi()` function for numeric predictions and health warnings.

## Dataset
The project utilizes the [Air Quality Data in India dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) authored by Rohan Rao.
* **Using Kaggle API (Recommended)**: The script automatically attempts to download the dataset via the Kaggle API if it isn't found locally. Ensure you have your Kaggle authentication token configured (`~/.kaggle/kaggle.json`).
* **Local Backup**: Alternatively, you can download `city_day.csv` manually and place it in the same directory as the execution script.

## Model Comparison

| Model               | MAE   | RMSE  | R² Score | MAPE (%) |
|----------------------|-------|-------|----------|----------|
| Baseline (Mean)      | —     | —     | 0.0000   | —        |
| Linear Regression    | —     | —     | ~0.60    | —        |
| Decision Tree        | —     | —     | ~0.90    | —        |
| **Random Forest**    | —     | —     | **~0.95**| —        |

> *Run `python aqi_prediction.py` to get the actual numbers for your dataset.*

## Machine Learning Models Used
1. **Baseline (DummyRegressor)** — always predicts the mean AQI
2. **Linear Regression**
3. **Decision Tree Regressor**
4. **Random Forest Regressor** (Chosen for the final model)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd AirGuard
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the ML Pipeline
```bash
python aqi_prediction.py
```
This will automatically:
1. Load/download the data.
2. Output comparison metrics (with baseline, MAE, RMSE, R², MAPE) in the terminal.
3. Generate and save EDA graphs, residual plot, SHAP summary, and feature importance to `outputs/`.
4. Output an example prediction with SHAP contributions.

### Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Opens an interactive dashboard with:
- City AQI trend over time
- Predicted vs actual AQI
- Anomaly detection panel
- AQI prediction with health advisory

### Enable AI Health Advisory (Optional)
Set the Groq API key as an environment variable:
```bash
# Linux/macOS
export LLM_API_KEY="your-groq-api-key"

# Windows (PowerShell)
$env:LLM_API_KEY="your-groq-api-key"

# Windows (CMD)
set LLM_API_KEY=your-groq-api-key
```
Without the key, the system uses a template-based fallback that works fully offline.

## Example Prediction Output

```text
Predicted AQI: 165
Category: Unhealthy for Sensitive Groups
Recommendation: Sensitive groups should reduce outdoor activity.

SHAP Contributions (per pollutant):
  PM2.5     : +32.45
  PM10      : +15.20
  CO        : +4.80
  NO2       : -2.10
  SO2       : -0.90
  O3        : -0.55
```

## Project Structure

```text
AirGuard/
│
├── aqi_prediction.py       # Main ML pipeline: data, training, evaluation, SHAP, prediction
├── anomaly_detection.py    # Isolation Forest anomaly detection
├── genai_advisory.py       # GenAI health advisory (Gemini + offline fallback)
├── app.py                  # Streamlit dashboard
├── export_model.py         # Export model to joblib for API
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation (this file)
├── city_day.csv            # Dataset
├── LICENSE                 # Open-source MIT License
├── api/                    # Vercel serverless API
│   ├── index.py
│   ├── aqi_model.joblib
│   └── requirements.txt
└── outputs/                # (Generated) Saved plots and visualizations
```

## Future Improvements
- **Real-Time Data**: Connect the predictor to a live IoT sensor API for real-time local monitoring.
- **Hyperparameter Tuning**: Introduce `GridSearchCV` to optimize the Random Forest Regressor for even higher accuracy.
- **Multi-city Comparison**: Add side-by-side city comparisons in the dashboard.
