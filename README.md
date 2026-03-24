# AirGuard – Air Quality Prediction and Health Advisory System

## Project Overview
AirGuard is a Machine Learning project to predict Air Quality Index (AQI) based on pollutant levels. It includes data processing, exploratory data analysis (EDA), model comparison, and a prediction function.

## Problem Statement
The goal of this system is to accurately map multivariate pollutant inputs to a standardized Air Quality Index (AQI) and provide corresponding health recommendations.

## Features
- **Data Processing**: Automatically loads, cleans, and handles missing values.
- **Exploratory Data Analysis (EDA)**: Generates visual insights like histograms, heatmaps, and scatter plots.
- **Model Training & Comparison**: Tests Linear Regression, Decision Tree, and Random Forest models.
- **Model Insights**: Analyzes and plots feature importance for AQI predictions.
- **Automated Kaggle Integration**: Can pull the dataset securely via the Kaggle API.
- **Actionable Prediction**: Offers a reusable `predict_aqi()` function for numeric predictions and health warnings.

## Dataset
The project utilizes the [Air Quality Data in India dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) authored by Rohan Rao.
* **Using Kaggle API (Recommended)**: The script automatically attempts to download the dataset via the Kaggle API if it isn't found locally. Ensure you have your Kaggle authentication token configured (`~/.kaggle/kaggle.json`).
* **Local Backup**: Alternatively, you can download `city_day.csv` manually and place it in the same directory as the execution script.

## Machine Learning Models Used
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor** (Chosen for the final model)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd AirGuard
   ```

2. **Install required dependencies:**
   Make sure you have Python installed, then use the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the full machine learning pipeline with a single command:
```bash
python aqi_prediction.py
```
This will automatically:
1. Load/download the data.
2. Output comparison metrics for the different models in the terminal.
3. Generate and save EDA graphs to an `outputs/` folder.
4. Output an example prediction at the end.

## Example Prediction Output

When utilizing the `predict_aqi()` function on a new set of sensor data:

```text
Predicted AQI: 165
Category: Unhealthy for Sensitive Groups
Recommendation: Sensitive groups should reduce outdoor activity. The general public is less likely to be affected.
```

## Project Structure

```text
AirGuard/
│
├── aqi_prediction.py     # Main Python script for training, evaluating, and predicting
├── requirements.txt      # Python dependencies needed to run the project
├── README.md             # Project documentation (this file)
├── LICENSE               # Open-source MIT License
└── outputs/              # (Generated) Folder containing saved plots and visualizations
```

## Future Improvements
- **Web Interface**: Wrap the `predict_aqi()` function into a lightweight Flask or Streamlit web application.
- **Real-Time Data**: Connect the predictor to a live IoT sensor API for real-time local monitoring.
- **Hyperparameter Tuning**: Introduce `GridSearchCV` to optimize the Random Forest Regressor for even higher accuracy.
