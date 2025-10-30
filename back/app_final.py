# --- Import necessary libraries ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import ast
from haversine import haversine, Unit
from twilio.rest import Client

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
from flask_cors import CORS

# Apply CORS to all /api/* routes
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)




# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "xxxxxxxxxxxxxxxxxx"
TWILIO_PHONE_NUMBER = "xxxxxxxxxx"
BLOOD_BANK_PHONE_NUMBER = "xxxxxxxx"

# --- Load ML Model (if available) ---
try:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    with open('model_vitals.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('scaler_vitals.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Model load error: {e}")
    ml_model = None
    scaler = None

# --- Load blood bank data ---
try:
    blood_banks_df = pd.read_csv('blood_banks_formatted.csv')
    blood_banks_df = blood_banks_df.rename(columns={'O': 'O+', 'A': 'A+', 'B': 'B+', 'AB': 'AB+'})
    blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
    for btype in blood_types:
        if btype not in blood_banks_df.columns:
            blood_banks_df[btype] = 0
        blood_banks_df[btype] = pd.to_numeric(blood_banks_df[btype], errors='coerce').fillna(0).astype(int)
    blood_banks_df['expiry_dates'] = blood_banks_df['expiry_dates'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    blood_banks_df['lat'] = pd.to_numeric(blood_banks_df['lat'], errors='coerce')
    blood_banks_df['lon'] = pd.to_numeric(blood_banks_df['lon'], errors='coerce')
except Exception as e:
    print(f"Error loading blood bank data: {e}")
    blood_banks_df = pd.DataFrame()

# --- Helper Function: Predict PPH Risk ---
def predict_pph(data):
    try:
        if not ml_model or not scaler:
            return {"error": "Model not loaded"}, 500
        input_data = np.array([
            float(data.get("HeartRate", 0)),
            float(data.get("RespiratoryRate", 0)),
            float(data.get("SystolicBP", 0)),
            float(data.get("DiastolicBP", 0)),
            float(data.get("OxygenSaturation", 0))
        ]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = ml_model.predict(input_scaled)
        risk = int(prediction[0])
        return {"pph_risk": risk}, 200
    except Exception as e:
        return {"error": str(e)}, 500

# --- API Route: Predict PPH ---
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result, status = predict_pph(data)
        return jsonify(result), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- API Route: Find Nearest Blood Bank ---
@app.route("/api/find_blood_bank", methods=["POST"])
def find_blood_bank():
    try:
        data = request.get_json()
        user_lat = float(data.get("lat"))
        user_lon = float(data.get("lon"))
        blood_group = data.get("blood_group")
        if blood_banks_df.empty:
            return jsonify({"error": "Blood bank data not loaded"}), 500
        def compute_distance(row):
            return haversine((user_lat, user_lon), (row["lat"], row["lon"]), unit=Unit.KILOMETERS)
        blood_banks_df["distance_km"] = blood_banks_df.apply(compute_distance, axis=1)
        nearby_banks = blood_banks_df[blood_banks_df[blood_group] > 0]
        nearby_banks = nearby_banks.sort_values("distance_km").head(5)
        result = nearby_banks.to_dict(orient="records")
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- API Route: Send SMS Notification ---
@app.route("/api/send_sms", methods=["POST"])
def send_sms():
    try:
        data = request.get_json()
        message_body = data.get("message", "Alert from Hemosync")
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            from_=TWILIO_PHONE_NUMBER,
            to=BLOOD_BANK_PHONE_NUMBER,
            body=message_body
        )
        return jsonify({"status": "SMS sent successfully", "sid": message.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Added Missing Function from app_1.py ---
@app.route("/api/patients", methods=["GET"])
def get_patients():
    try:
        patients = [
            {"id": 1, "name": "Alice", "age": 32, "blood_group": "A+"},
            {"id": 2, "name": "Bob", "age": 45, "blood_group": "B+"},
            {"id": 3, "name": "Charlie", "age": 29, "blood_group": "O-"}
        ]
        return jsonify({"patients": patients}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(debug=True)
