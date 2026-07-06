# -*- coding: utf-8 -*-
"""
AirGuard – Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="AirGuard Dashboard", page_icon="🌬️", layout="wide")

# --- Load project modules (triggers init on first import) ---
from aqi_prediction import (
    init, features, target, predict_aqi, get_shap_explanation, get_aqi_category
)
from anomaly_detection import detect_anomalies
from genai_advisory import generate_advisory

# Initialize the ML pipeline (only runs once)
init()
from aqi_prediction import data as raw_data, df as cleaned_df, best_model, X_test, y_test

# Check which optional columns exist
HAS_CITY = 'City' in raw_data.columns
HAS_DATE = 'Date' in raw_data.columns


# ==========================================
# Data Preparation
# ==========================================
@st.cache_data
def prepare_full_data():
    """Prepare data for dashboard — handles datasets with or without City/Date."""
    df = raw_data.copy()
    if HAS_DATE:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
    df = df.dropna(subset=['AQI'])
    df[features] = df[features].fillna(df[features].mean())
    return df


@st.cache_data
def get_anomaly_data():
    """Run anomaly detection on the full dataset."""
    df = prepare_full_data()
    return detect_anomalies(df)


full_df = prepare_full_data()


# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("🌬️ AirGuard")
st.sidebar.markdown("Air Quality Prediction & Advisory")
st.sidebar.markdown("---")

# City selector (only if City column exists)
selected_city = None
if HAS_CITY:
    cities = sorted(full_df['City'].dropna().unique())
    selected_city = st.sidebar.selectbox(
        "Select City", cities,
        index=cities.index("Delhi") if "Delhi" in cities else 0
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### Predict AQI")
pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 45.0, 1.0)
pm10 = st.sidebar.slider("PM10", 0.0, 600.0, 120.0, 1.0)
no2 = st.sidebar.slider("NO2", 0.0, 250.0, 30.0, 1.0)
so2 = st.sidebar.slider("SO2", 0.0, 200.0, 15.0, 1.0)
o3 = st.sidebar.slider("O3", 0.0, 300.0, 40.0, 1.0)
co = st.sidebar.slider("CO", 0.0, 30.0, 1.2, 0.1)


# ==========================================
# Panel 1: AQI Distribution / Trend
# ==========================================
title = f"📊 AQI Dashboard — {selected_city}" if selected_city else "📊 AQI Dashboard"
st.title(title)

# Filter by city if available
display_df = full_df[full_df['City'] == selected_city] if HAS_CITY and selected_city else full_df
if HAS_DATE:
    display_df = display_df.sort_values('Date')

col1, col2 = st.columns(2)

with col1:
    if HAS_DATE:
        st.subheader("AQI Trend Over Time")
        if not display_df.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(display_df['Date'], display_df['AQI'], color='steelblue', linewidth=0.8, alpha=0.8)
            ax1.set_xlabel("Date")
            ax1.set_ylabel("AQI")
            ax1.set_title("AQI Trend" + (f" — {selected_city}" if selected_city else ""))
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("No data available.")
    else:
        st.subheader("AQI Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.hist(display_df['AQI'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax1.set_xlabel("AQI")
        ax1.set_ylabel("Frequency")
        ax1.set_title("AQI Distribution")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)


# ==========================================
# Panel 2: Predicted vs Actual
# ==========================================
with col2:
    st.subheader("Predicted vs Actual AQI")
    plot_data = display_df if len(display_df) > 10 else full_df
    X_plot = plot_data[features]
    y_plot = plot_data['AQI']
    y_plot_pred = best_model.predict(X_plot)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.scatter(y_plot, y_plot_pred, alpha=0.4, color='purple', s=10)
    lims = [min(y_plot.min(), y_plot_pred.min()), max(y_plot.max(), y_plot_pred.max())]
    ax2.plot(lims, lims, 'r--', linewidth=1)
    ax2.set_xlabel("Actual AQI")
    ax2.set_ylabel("Predicted AQI")
    ax2.set_title("Predicted vs Actual" + (f" — {selected_city}" if selected_city and len(display_df) > 10 else ""))
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")

# ==========================================
# Panel 3: Anomaly Detection
# ==========================================
st.subheader("🚨 Anomaly Detection — Pollution Spikes")

anomaly_df = get_anomaly_data()
if HAS_CITY and selected_city:
    city_anomalies = anomaly_df[anomaly_df['City'] == selected_city]
else:
    city_anomalies = anomaly_df

if not city_anomalies.empty:
    n_anomalies = city_anomalies['is_anomaly'].sum()
    st.metric("Anomalies Flagged", f"{n_anomalies} / {len(city_anomalies)}")

    fig3, ax3 = plt.subplots(figsize=(14, 4))
    normal = city_anomalies[~city_anomalies['is_anomaly']]
    spikes = city_anomalies[city_anomalies['is_anomaly']]

    x_axis = 'Date' if HAS_DATE else None

    if x_axis:
        ax3.plot(normal[x_axis], normal['AQI'], color='steelblue', linewidth=0.5, alpha=0.6, label='Normal')
        ax3.scatter(spikes[x_axis], spikes['AQI'], color='red', s=25, zorder=5, label='Anomaly')
        ax3.set_xlabel("Date")
        plt.xticks(rotation=45)
    else:
        ax3.scatter(normal.index, normal['AQI'], s=5, alpha=0.3, color='steelblue', label='Normal')
        ax3.scatter(spikes.index, spikes['AQI'], s=20, alpha=0.8, color='red', label='Anomaly', zorder=5)
        ax3.set_xlabel("Sample Index")

    ax3.set_ylabel("AQI")
    ax3.set_title("Pollution Spikes (Isolation Forest)")
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    if n_anomalies > 0:
        with st.expander("View flagged records"):
            show_cols = [c for c in ['Date', 'City', 'AQI'] + features if c in spikes.columns]
            st.dataframe(spikes[show_cols].head(20))
else:
    st.info("No data available for anomaly detection.")

st.markdown("---")

# ==========================================
# Panel 4: Prediction + Advisory
# ==========================================
st.subheader("🔮 AQI Prediction & Health Advisory")

pred_aqi, category, recommendation = predict_aqi(pm25, pm10, no2, so2, o3, co)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted AQI", f"{int(round(pred_aqi))}")
c2.metric("Category", category)
c3.metric("Health Risk", "Low" if pred_aqi <= 100 else "Medium" if pred_aqi <= 200 else "High")

# SHAP explanation
shap_input = dict(zip(features, [pm25, pm10, no2, so2, o3, co]))
shap_contribs = get_shap_explanation(shap_input)

# GenAI advisory
advisory_input = {
    "aqi": pred_aqi,
    "category": category,
    "city": selected_city or "Unknown",
    "date": str(pd.Timestamp.now().date()),
    "shap_contributions": shap_contribs,
}
advisory = generate_advisory(advisory_input)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🏥 Health Advisory")
    st.info(advisory["health_advisory"])

with col_b:
    st.markdown("#### 🔬 What's Driving the AQI")
    st.info(advisory["aqi_explanation"])

    st.markdown("**SHAP Contributions:**")
    shap_df = pd.DataFrame([
        {"Pollutant": k, "Contribution": f"{v:+.2f}"} for k, v in shap_contribs.items()
    ])
    st.table(shap_df)
