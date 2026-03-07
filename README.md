# AQI Prediction Project

This project predicts the Air Quality Index (AQI) based on various pollutant levels leveraging Machine Learning models. The existing codebase has been extended into a more comprehensive data science project while remaining accessible and suitable for student-level learning.

## Features

- **Data Processing**: Loads the `city_day.csv` dataset, handles missing values, and isolates the target/feature sets. 
- **Exploratory Data Analysis (EDA)**: Includes visualizations such as:
  - AQI Distribution Histogram
  - Correlation Heatmap between Pollutants and AQI
  - Scatter plots of Pollutants versus AQI
- **Model Training & Comparison**: 
  - Validates performance using Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
  - Generates clear comparative metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
- **Model Insights**: Features a visualization displaying the most influential pollutants contributing to the AQI score (using Random Forest feature importance).
- **Actual vs. Predicted**: Plotted comparison of expected and actual AQI values to visually assess model performance.
- **Actionable Prediction**: Offers a reusable `predict_aqi()` Python function that outputs the predicted AQI, category classification, and generic health recommendation.

## AQI Categories

- **Good** (0-50): Air quality is satisfactory.
- **Moderate** (51-100): Air quality is acceptable, but might affect highly sensitive groups.
- **Unhealthy for Sensitive Groups** (101-150): Members of sensitive groups may experience health effects.
- **Unhealthy** (151-200): General public may experience health effects.
- **Very Unhealthy** (201-300): Health alert; risk of effects is increased for everyone.
- **Hazardous** (301+): Health warning of emergency conditions.

## Setup & General Usage

1. **Environment Requirements**: Ensure you have Python installed, along with the necessary data science libraries.
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kaggle
   ```
2. **Dataset**: 
   - **Using Kaggle API (Recommended)**: The script automatically attempts to download the dataset via the Kaggle API if it isn't found locally. Ensure you have your Kaggle authentication token configured (`~/.kaggle/kaggle.json` or `C:\Users\<User>\.kaggle\kaggle.json` on Windows). 
   - **Local Backup**: If the API approach fails, simply download the [Air Quality Data in India dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) manually. Extract the zip file and place `city_day.csv` in the same directory as the execution script. 
   - **Colab**: The script also handles Google Colab environments by looking for `/content/city_day.csv`.
3. **Execution**: Run the Python file directly or include parts in Jupyter Notebooks / Google Colab to see metrics and EDA graphs.

```bash
python aqi_prediction.py
```

## Making Predictions
Within your notebook or standard Python script you can call the `predict_aqi()` function directly with the required pollutant features (PM2.5, PM10, NO2, SO2, O3, CO):

```python
from aqi_prediction import predict_aqi

# Sample query: pm2.5=45.0, pm10=120.0, no2=30.0, so2=15.0, o3=40.0, co=1.2
pred_aqi, category, recommendation = predict_aqi(45.0, 120.0, 30.0, 15.0, 40.0, 1.2)
print(f"Predicted AQI: {pred_aqi:.2f} | Category: {category}")
print(f"Recommendation: {recommendation}")
```

## Structure
- `aqi_prediction.py`: The single end-to-end Python file containing all modular steps (Loading, Cleaning, EDA, Model Comparison, Reusable Predictor function).
- `city_day.csv`: The dataset utilized for training models (Make sure this is present!).
- `README.md`: Provided project overview documentation.
