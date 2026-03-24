from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model when Vercel spins up the serverless function
MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqi_model.joblib")
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "Air quality is satisfactory."
    elif aqi_value <= 100:
        return "Moderate", "Air quality is acceptable."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "Sensitive groups should reduce outdoor activity."
    elif aqi_value <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased."
    else:
        return "Hazardous", "Health warning of emergency conditions."

@app.route('/api', methods=['GET'])
def home():
    if model is None:
        return jsonify({"error": "Model not found. Please train and save the model first."}), 500
    return jsonify({"message": "AirGuard API is running! Use the /predict endpoint."})

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not found."}), 500
        
    try:
        # Get data from query params (GET) or JSON body (POST)
        if request.method == 'POST':
            data = request.get_json()
        else:
            data = request.args
            
        # Extract features
        features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
        input_values = []
        
        for feature in features:
            val = data.get(feature)
            if val is None:
                return jsonify({"error": f"Missing required parameter: {feature}"}), 400
            input_values.append(float(val))
            
        # Predict
        input_data = pd.DataFrame([input_values], columns=features)
        predicted_value = float(model.predict(input_data)[0])
        category, recommendation = get_aqi_category(predicted_value)
        
        return jsonify({
            "predicted_aqi": round(predicted_value, 2),
            "category": category,
            "recommendation": recommendation
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Vercel requires the app variable to be exposed
if __name__ == '__main__':
    app.run(debug=True)
