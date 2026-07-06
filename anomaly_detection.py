# -*- coding: utf-8 -*-
"""
AirGuard – Anomaly Detection Module
Flags pollution spikes / anomalous readings using Isolation Forest.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

POLLUTANT_FEATURES = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']


def detect_anomalies(df, contamination=0.05):
    """
    Flag anomalous pollution readings using Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the pollutant feature columns.
    contamination : float
        Expected proportion of anomalies (default 5%).

    Returns
    -------
    pd.DataFrame
        Copy of input with added `is_anomaly` (bool) and `anomaly_score` (float) columns.
    """
    result = df.copy()

    # Use only pollutant columns that exist in the dataframe
    available = [c for c in POLLUTANT_FEATURES if c in result.columns]
    if not available:
        result['is_anomaly'] = False
        result['anomaly_score'] = 0.0
        return result

    X = result[available].fillna(result[available].mean())

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)

    result['is_anomaly'] = preds == -1
    result['anomaly_score'] = scores
    return result


def plot_anomalies(df, save_path=None):
    """
    Save a plot highlighting anomalous spikes. Uses Date if available, else row index.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `detect_anomalies()` — must have `is_anomaly` and `AQI` columns.
    save_path : str, optional
        Path to save the figure. Defaults to outputs/anomaly_spikes.png.
    """
    if save_path is None:
        save_path = os.path.join(output_dir, 'anomaly_spikes.png')

    if 'AQI' not in df.columns or 'is_anomaly' not in df.columns:
        print("Cannot plot anomalies: missing AQI or is_anomaly column.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    normal = df[~df['is_anomaly']]
    anomalies = df[df['is_anomaly']]

    ax.scatter(normal.index, normal['AQI'], s=5, alpha=0.3, color='steelblue', label='Normal')
    ax.scatter(anomalies.index, anomalies['AQI'], s=20, alpha=0.8, color='red', label='Anomaly', zorder=5)
    ax.set_title("Pollution Anomaly Detection (Isolation Forest)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("AQI")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Anomaly plot saved to {save_path}")


if __name__ == "__main__":
    from aqi_prediction import init, data, features, target
    init()
    from aqi_prediction import data as raw_data

    sample = raw_data[features + [target]].copy().dropna(subset=[target])
    sample[features] = sample[features].fillna(sample[features].mean())

    result = detect_anomalies(sample)
    print(f"Total anomalies flagged: {result['is_anomaly'].sum()} / {len(result)}")
    plot_anomalies(result)
