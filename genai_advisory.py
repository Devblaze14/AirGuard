# -*- coding: utf-8 -*-
"""
AirGuard – GenAI Advisory Layer
Generates natural-language health advisories using an LLM (Groq),
with a graceful offline fallback.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()



def _get_fallback_advisory(aqi, category, shap_contributions=None):
    """
    Template-based fallback advisory when no LLM API key is available.
    Always works offline.
    """
    # Build a simple explanation from SHAP contributions
    explanation = ""
    if shap_contributions:
        top = list(shap_contributions.items())[:3]
        drivers = ", ".join(f"{feat} ({val:+.1f})" for feat, val in top)
        explanation = f"The main pollutants driving today's AQI are: {drivers}."
    else:
        explanation = "Pollutant contribution data is not available."

    # Category-based health advisory
    advisories = {
        "Good": {
            "sensitive": "No precautions needed. Enjoy outdoor activities.",
            "general": "Air quality is excellent. No health risks.",
        },
        "Moderate": {
            "sensitive": "Unusually sensitive individuals should consider reducing prolonged outdoor exertion.",
            "general": "Air quality is acceptable for most people.",
        },
        "Unhealthy for Sensitive Groups": {
            "sensitive": "People with respiratory or heart conditions, children, and the elderly should limit outdoor activity.",
            "general": "General public is less likely to be affected but should be aware.",
        },
        "Unhealthy": {
            "sensitive": "Avoid prolonged outdoor exertion. Keep windows closed and use air purifiers if available.",
            "general": "Everyone may begin to experience health effects. Reduce outdoor activity.",
        },
        "Very Unhealthy": {
            "sensitive": "Stay indoors. Use N95 masks if you must go outside. Seek medical attention if symptomatic.",
            "general": "Health alert — everyone should significantly reduce outdoor exposure.",
        },
        "Hazardous": {
            "sensitive": "Remain indoors with air filtration. Seek immediate medical help for any symptoms.",
            "general": "Emergency conditions — everyone should avoid all outdoor activity.",
        },
    }

    advice = advisories.get(category, advisories["Moderate"])

    health_advisory = (
        f"🏥 Health Advisory (AQI {int(round(aqi))} — {category}):\n"
        f"  • Sensitive groups: {advice['sensitive']}\n"
        f"  • General public: {advice['general']}"
    )

    return {
        "health_advisory": health_advisory,
        "aqi_explanation": explanation,
        "source": "template",
    }


def _call_groq(prompt, api_key):
    """Call Groq API and return the text response."""
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content


def generate_advisory(prediction):
    """
    Generate a natural-language health advisory and AQI explanation.

    Parameters
    ----------
    prediction : dict
        Required keys: aqi, category.
        Optional keys: city, date, shap_contributions.

    Returns
    -------
    dict with keys: health_advisory, aqi_explanation, source ("groq" or "template").
    """
    aqi = prediction.get("aqi", 0)
    category = prediction.get("category", "Moderate")
    city = prediction.get("city", "Unknown")
    date = prediction.get("date", "today")
    shap_contributions = prediction.get("shap_contributions", {})

    api_key = os.environ.get("LLM_API_KEY", "")

    # --- Fallback if no API key ---
    if not api_key:
        return _get_fallback_advisory(aqi, category, shap_contributions)

    # --- Build the LLM prompt ---
    shap_text = ""
    if shap_contributions:
        lines = [f"  - {feat}: {val:+.2f}" for feat, val in shap_contributions.items()]
        shap_text = "\n".join(lines)
    else:
        shap_text = "  (no SHAP data available)"

    prompt = f"""You are AirGuard, an air-quality health advisor.

Given the following prediction data, produce EXACTLY two sections:

1. **Health Advisory**: Separate guidance for (a) sensitive groups (children, elderly, people with respiratory/heart conditions) and (b) the general public. Be specific and actionable.

2. **What's Driving Today's AQI**: A plain-English explanation referencing the actual top pollutants from the SHAP contributions below. Do NOT invent pollutant data — use only what is provided.

Data:
- City: {city}
- Date: {date}
- Predicted AQI: {int(round(aqi))}
- AQI Category: {category}
- SHAP pollutant contributions (positive = increases AQI, negative = decreases):
{shap_text}

Keep the response concise (under 200 words). Do not use markdown headers — just plain text with bullet points."""

    try:
        text = _call_groq(prompt, api_key)
        # Split into two sections heuristically
        parts = text.split("What's Driving", 1)
        if len(parts) == 2:
            health = parts[0].strip()
            explanation = "What's Driving" + parts[1].strip()
        else:
            health = text.strip()
            explanation = ""

        return {
            "health_advisory": health,
            "aqi_explanation": explanation if explanation else f"AQI is {int(round(aqi))} ({category}).",
            "source": "groq",
        }
    except Exception as e:
        print(f"LLM call failed ({e}), falling back to template.")
        return _get_fallback_advisory(aqi, category, shap_contributions)


if __name__ == "__main__":
    # Quick test
    test_pred = {
        "aqi": 185,
        "category": "Unhealthy",
        "city": "Delhi",
        "date": "2024-01-15",
        "shap_contributions": {"PM2.5": 42.3, "PM10": 18.7, "CO": 5.1, "NO2": -2.4, "SO2": -1.0, "O3": -0.8},
    }
    result = generate_advisory(test_pred)
    print(f"Source: {result['source']}")
    print(result['health_advisory'])
    print()
    print(result['aqi_explanation'])
