# --- Import necessary libraries ---
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_bytes
import pytesseract
import re
import io
import os
import traceback
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import ast
from haversine import haversine, Unit
from twilio.rest import Client
import speech_recognition as sr
import time
import threading
from datetime import datetime
import warnings
import queue

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "ngrok-skip-browser-warning"]
    }
})


# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your Twilio SID
TWILIO_AUTH_TOKEN = "xxxxxxxxxxxxxxxxxxxx"               # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "xxxxxxxxxxxx"                       # Replace with your Twilio phone number (e.g., "+15551234567")
BLOOD_BANK_PHONE_NUMBER = "xxxxxxxxxxxxxx"                  # Replace with the blood bank's phone number (e.g., "+919876543210")

# --- Microphone Monitoring Configuration ---
TARGET_PHRASE = "blood blood blood"
PHRASE_DETECTION_WINDOW = 5  # seconds to capture each audio chunk

# Add these global variables at the top with other configurations
monitoring_active = False
audio_queue = queue.Queue()
latest_alert = None
alerts_queue = queue.Queue()

# --- Load ML Model (if available) ---
try:
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

    with open('model_vitals.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('scaler_vitals.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('vital_signs_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    MODEL_LOADED = True
    print("✅ ML Model loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️ ML Model not found. Using rule-based risk assessment. error: {e}")

# --- Load Blood Bank Data ---
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

    BLOOD_BANKS_LOADED = True
    print("✅ Blood bank data loaded successfully!")
except Exception as e:
    BLOOD_BANKS_LOADED = False
    blood_banks_df = None
    print(f"⚠️ Blood bank data not found: {e}")

# --- Load Patient Data ---
try:
    patients_df = pd.read_csv('patients.csv')
    patients_df['name'] = patients_df['name'].fillna(lambda x: f'Patient {x.index + 1}')
    PATIENTS_LOADED = True
    print("✅ Patient data loaded successfully!")
except Exception as e:
    PATIENTS_LOADED = False
    patients_df = None
    print(f"⚠️ Patient data not found: {e}")

# --- OCR Function ---
def extract_text_from_file(uploaded_file):
    """Extract text from PDF or image using OCR."""
    try:
        filename = uploaded_file.filename.lower()
        file_bytes = uploaded_file.read()
        if not file_bytes:
            raise ValueError("Empty file received")
        text = ""
        if filename.endswith(".pdf"):
            pages = convert_from_bytes(file_bytes, size=(1500, None), dpi=200)
            for page in pages:
                page = page.convert('L')
                text += pytesseract.image_to_string(page, config='--psm 6')
        else:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            width, height = image.size
            max_width = 1500
            if width > max_width:
                aspect_ratio = height / width
                new_height = int(max_width * aspect_ratio)
                image = image.resize((max_width, new_height), Image.LANCZOS)
            image = image.convert('L')
            text = pytesseract.image_to_string(image, config='--psm 6')
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        traceback.print_exc()
        raise Exception(f"OCR processing failed: {str(e)}")
    return text

# --- Extraction Logic ---
def extract_parameters(text):
    """Extract key PPH parameters using regex."""
    results = {}
    text_normalized = ' '.join(text.split())
    patterns = {
        "Patient Name": r'(?i)(?:Patient\s*Name|Name|Mother\s*Name)\s*[:\s]*([\w\s]+?)(?=\n|Patient|Age|DOB|$)',
        "Patient ID": r'(?i)(?:Patient\s*ID|ID\s*No|Registration\s*No|MRN)\s*[:\s]*([A-Z0-9-]+)',
        "Blood Type": r'(?i)(?:Blood\s*Type|Blood\s*Group)\s*[:\s]*([ABO]+[+-])',
        "Hospital": r'(?i)(?:Hospital|Facility|Health\s*Center|Wellness\s*Hospital)\s*[:\s]*([\w\s,]+?)(?=\n|Patient|$)',
        "Expected Delivery Date": r'(?i)(?:EDD|Expected\s*Delivery\s*Date|Due\s*Date)\s*[:\s]*([\d/-]+)',
        "Age (years)": r'(?i)Age\s*[:\s]*(\d+)\s*years?',
        "Parity": r'(?i)Parity\s*[:\s]*[Gg](\d+)[Pp](\d+)',
        "History of PPH": r'(?i)History\s*of\s*(?:Postpartum\s*)?Hemorrhage\s*[:\s]*(Yes|No)',
        "History of Previous Cesarean": r'(?i)(?:Previous\s*(?:Cesarean|Caesarean|C[- ]?section)|History\s*of\s*(?:CS|C[- ]?section))\s*[:\s]*(Yes|No)',
        "Pre-pregnancy BMI": r'(?i)Pre[- ]?Pregnancy\s*BMI\s*[:\s]*([\d.]+)\s*(?:kg/m[²2])?',
        "Hemoglobin (g/dL)": r'(?i)Hemoglobin\s*(?:level|Level)?\s*[:\s]*([\d.]+)\s*(?:g/dL)?',
        "Platelet Count (x10^3/μL)": r'(?i)Platelet\s*Count\s*[:\s]*([\d,]+)\s*(?:/mm[³3])?',
        "Systolic Blood Pressure (mmHg)": r'(?i)Blood\s*Pressure\s*[:\s]*(\d+)/\d+\s*mmHg',
        "Diastolic Blood Pressure (mmHg)": r'(?i)Blood\s*Pressure\s*[:\s]*\d+/(\d+)\s*mmHg',
        "Heart Rate (bpm)": r'(?i)Heart\s*Rate\s*[:\s]*(\d+)\s*bpm',
        "Placenta Previa": r'(?i)Placenta\s*Previa\s*[:\s]*(Yes|No)',
        "Multiple Pregnancy": r'(?i)Multiple\s*Pregnancy\s*[:\s]*(?:sal\s*[:\s]*)?(Yes|No)',
        "Estimated Fetal Weight (kg)": r'(?i)Estimated\s*Fetal\s*Weight\s*[:\s]*([\d.]+)\s*kg',
        "Gestational Age (weeks)": r'(?i)Gestational\s*Age\s*[:\s]*(\d+)\s*weeks?',
        "Mode of Delivery": r'(?i)Mode\s*of\s*Delivery\s*[:\s]*(C[- ]?Section|Cesarean|Caesarean|Vaginal|Normal)',
        "Induced Labor": r'(?i)Induced\s*Labor\s*[:\s]*(Yes|No)',
        "Anesthesia Type": r'(?i)Anesthesia\s*type\s*[:\s]*(General|Spinal|Epidural|Local)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            if key == "Parity":
                results[key] = match.group(2).strip()
            elif key == "Platelet Count (x10^3/μL)":
                results[key] = match.group(1).replace(',', '').strip()
            elif key == "Mode of Delivery":
                value = match.group(1).strip()
                results[key] = "C-Section" if re.search(r'(?i)c[- ]?section|cesarean|caesarean', value) else value
            else:
                results[key] = match.group(1).strip()
    # Special handling
    if "Multiple Pregnancy" not in results:
        sal_match = re.search(r'(?i)sal\s*[:\s]*(Yes|No)', text)
        if sal_match:
            results["Multiple Pregnancy"] = sal_match.group(1)
    if "Gestational Age (weeks)" not in results:
        ga_match = re.search(r'(?i)(?:GA|Gestational\s*Age)\s*[:\s]*(\d+)\s*w(?:eeks?|ks?)', text)
        if ga_match:
            results["Gestational Age (weeks)"] = ga_match.group(1)
    if "Systolic Blood Pressure (mmHg)" not in results or "Diastolic Blood Pressure (mmHg)" not in results:
        bp_match = re.search(r'(?i)(?:BP|Blood\s*Pressure)\s*[:\s]*(\d+)/(\d+)', text)
        if bp_match:
            results["Systolic Blood Pressure (mmHg)"] = bp_match.group(1)
            results["Diastolic Blood Pressure (mmHg)"] = bp_match.group(2)
    return results

# --- Risk Calculation ---
def calculate_pph_risk(findings):
    """Calculate PPH risk using ML model or rule-based approach."""
    def safe_float(value, default=0):
        try:
            return float(str(value).replace(',', ''))
        except:
            return default
    age = safe_float(findings.get('Age (years)', 0))
    systolic_bp = safe_float(findings.get('Systolic Blood Pressure (mmHg)', 0))
    diastolic_bp = safe_float(findings.get('Diastolic Blood Pressure (mmHg)', 0))
    heart_rate = safe_float(findings.get('Heart Rate (bpm)', 0))
    hemoglobin = safe_float(findings.get('Hemoglobin (g/dL)', 0))
    bmi = safe_float(findings.get('Pre-pregnancy BMI', 0))

    if MODEL_LOADED:
        try:
            age_risk = 1 if (age < 20 or age > 35) else 0
            hypertension = 1 if (systolic_bp >= 140 or diastolic_bp >= 90) else 0
            bp_product = systolic_bp * diastolic_bp
            hr_abnormal = 1 if (heart_rate < 60 or heart_rate > 100) else 0
            features_dict = {
                'age': age,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'heart_rate': heart_rate,
                'age_risk': age_risk,
                'hypertension': hypertension,
                'bp_product': bp_product,
                'hr_abnormal': hr_abnormal
            }
            if bmi > 0:
                features_dict['bmi'] = bmi
                features_dict['obesity'] = 1 if bmi >= 30 else 0
            feature_array = []
            for feat_name in feature_names:
                feature_array.append(features_dict.get(feat_name, 0))
            X = pd.DataFrame([feature_array], columns=feature_names)
            X_scaled = scaler.transform(X)
            risk_prob = float(ml_model.predict_proba(X_scaled)[0][1] * 100)
            feature_importance = dict(zip(feature_names, ml_model.feature_importances_))
            top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            return {
                'riskScore': round(risk_prob, 1),
                'riskLevel': 'HIGH' if risk_prob > 60 else 'MODERATE' if risk_prob > 30 else 'LOW',
                'modelType': 'ML',
                'topRiskFactors': [{'factor': f[0].replace('_', ' ').title(), 'importance': float(round(f[1], 3))} for f in top_factors]
            }
        except Exception as e:
            print(f"ML prediction error: {e}")
            traceback.print_exc()

    risk_score = 0
    risk_factors = []
    if age < 18:
        risk_score += 20
        risk_factors.append({'factor': 'Very Young Age (<18)', 'points': 20})
    elif age < 20 or age > 35:
        risk_score += 15
        risk_factors.append({'factor': 'Age Risk (<20 or >35)', 'points': 15})
    if systolic_bp >= 160 or diastolic_bp >= 110:
        risk_score += 25
        risk_factors.append({'factor': 'Severe Hypertension', 'points': 25})
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        risk_score += 20
        risk_factors.append({'factor': 'Hypertension', 'points': 20})
    if hemoglobin > 0 and hemoglobin < 11:
        risk_score += 20
        risk_factors.append({'factor': 'Anemia (Hb < 11 g/dL)', 'points': 20})
    elif hemoglobin > 0 and hemoglobin < 9:
        risk_score += 25
        risk_factors.append({'factor': 'Severe Anemia (Hb < 9 g/dL)', 'points': 25})
    if findings.get('History of PPH', '').lower() == 'yes':
        risk_score += 25
        risk_factors.append({'factor': 'Previous PPH History', 'points': 25})
    if findings.get('Placenta Previa', '').lower() == 'yes':
        risk_score += 30
        risk_factors.append({'factor': 'Placenta Previa', 'points': 30})
    if findings.get('Multiple Pregnancy', '').lower() == 'yes':
        risk_score += 15
        risk_factors.append({'factor': 'Multiple Pregnancy', 'points': 15})
    if findings.get('History of Previous Cesarean', '').lower() == 'yes':
        risk_score += 10
        risk_factors.append({'factor': 'Previous C-Section', 'points': 10})
    if bmi >= 35:
        risk_score += 15
        risk_factors.append({'factor': 'Severe Obesity (BMI ≥35)', 'points': 15})
    elif bmi >= 30:
        risk_score += 10
        risk_factors.append({'factor': 'Obesity (BMI ≥30)', 'points': 10})
    risk_score = min(risk_score, 100)
    return {
        'riskScore': risk_score,
        'riskLevel': 'HIGH' if risk_score > 60 else 'MODERATE' if risk_score > 30 else 'LOW',
        'modelType': 'Rule-Based',
        'topRiskFactors': sorted(risk_factors, key=lambda x: x['points'], reverse=True)[:5]
    }

# --- API Routes ---
@app.route("/api/ocr", methods=["POST", "OPTIONS"])
def ocr_api():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400
        text = extract_text_from_file(file)
        if not text or len(text.strip()) < 10:
            return jsonify({
                "error": "Could not extract meaningful text",
                "extractedText": text
            }), 400
        findings = extract_parameters(text)
        risk_assessment = calculate_pph_risk(findings)

        # Save patient data to CSV
        try:
            global patients_df, PATIENTS_LOADED
            if patients_df is None:
                patients_df = pd.DataFrame(columns=['patient_id', 'name', 'age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'hemoglobin', 'bmi', 'expected_delivery_date', 'status', 'trend', 'risk_score'])
                PATIENTS_LOADED = True

            # Generate new patient ID
            next_id = f'MH2024-{len(patients_df) + 1:03d}'
            
            # Get patient name from findings or generate one
            patient_name = findings.get('Patient Name', '').strip()
            if not patient_name:
                patient_name = f'Patient_{len(patients_df) + 1}'

            # Create new patient record
            new_patient = {
                'patient_id': next_id,
                'name': patient_name,
                'age': float(findings.get('Age (years)', 0)),
                'systolic_bp': float(findings.get('Systolic Blood Pressure (mmHg)', 0)),
                'diastolic_bp': float(findings.get('Diastolic Blood Pressure (mmHg)', 0)),
                'heart_rate': float(findings.get('Heart Rate (bpm)', 0)),
                'hemoglobin': float(findings.get('Hemoglobin (g/dL)', 0)),
                'bmi': float(findings.get('Pre-pregnancy BMI', 0)),
                'expected_delivery_date': findings.get('Expected Delivery Date', 'N/A'),
                'status': 'Monitoring',
                'trend': 'stable',
                'risk_score': risk_assessment['riskScore']
            }

            # Append to dataframe and save to CSV
            patients_df = pd.concat([patients_df, pd.DataFrame([new_patient])], ignore_index=True)
            patients_df.to_csv('patients.csv', index=False)
            print(f"✅ Patient {patient_name} saved to database")

        except Exception as save_error:
            print(f"⚠️ Error saving patient data: {save_error}")
            traceback.print_exc()

        return jsonify({
            "success": True,
            "documentType": "Maternal Health Report",
            "keyFindings": findings,
            "riskAssessment": risk_assessment,
            "extractedText": text[:2000],
            "metadata": {
                "version": "4.0-ML-Integrated",
                "parameters": list(findings.keys()),
                "filename": file.filename,
                "textLength": len(text),
                "mlModelLoaded": MODEL_LOADED,
                "patientSaved": True,
                "patientId": next_id
            }
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/api/risk-analysis", methods=["POST", "OPTIONS"])
def risk_analysis():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Convert boolean strings to actual boolean values
        for key in ['history_pph', 'placenta_previa', 'multiple_pregnancy', 'previous_cesarean']:
            if key in data:
                data[key] = str(data[key]).lower() == 'true'

        patient_data = {
            "Age (years)": float(data.get("age", 0)),
            "Systolic Blood Pressure (mmHg)": float(data.get("systolic_bp", 0)),
            "Diastolic Blood Pressure (mmHg)": float(data.get("diastolic_bp", 0)),
            "Heart Rate (bpm)": float(data.get("heart_rate", 0)),
            "Hemoglobin (g/dL)": float(data.get("hemoglobin", 0)),
            "Pre-pregnancy BMI": float(data.get("bmi", 0)),
            "History of PPH": "Yes" if data.get("history_pph") else "No",
            "Placenta Previa": "Yes" if data.get("placenta_previa") else "No",
            "Multiple Pregnancy": "Yes" if data.get("multiple_pregnancy") else "No",
            "History of Previous Cesarean": "Yes" if data.get("previous_cesarean") else "No",
            "Parity": str(data.get("parity", "0")),
            "Platelet Count (x10^3/μL)": str(data.get("platelet_count", "0")),
            "Gestational Age (weeks)": str(data.get("gestational_age", "0")),
            "Mode of Delivery": data.get("mode_of_delivery", "N/A"),
            "Estimated Fetal Weight (kg)": str(data.get("fetal_weight", "0"))
        }

        required_fields = ["Age (years)", "Systolic Blood Pressure (mmHg)", "Diastolic Blood Pressure (mmHg)", "Heart Rate (bpm)"]
        missing_fields = [field for field in required_fields if not patient_data[field]]
        
        if missing_fields:
            return jsonify({
                "success": False,
                "error": "Missing required fields",
                "missing": missing_fields,
                "required": required_fields
            }), 400

        risk_assessment = calculate_pph_risk(patient_data)

        response_data = {
            "success": True,
            "riskAssessment": risk_assessment,
            "metadata": {
                "version": "4.0-ML-Integrated",
                "mlModelLoaded": MODEL_LOADED,
                "fieldsProvided": len(patient_data),
                "requiredFields": required_fields
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "4.0-ML-Integrated",
        "mlModelLoaded": MODEL_LOADED
    }), 200

@app.route("/api/report", methods=["POST", "OPTIONS"])
def generate_report():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        payload = request.get_json(silent=True) or {}
        patient_data = payload.get("patientData") or {}
        risk_assessment = payload.get("riskAssessment")
        if not risk_assessment and patient_data:
            risk_assessment = calculate_pph_risk(patient_data)
        if not patient_data or not risk_assessment:
            return jsonify({
                "error": "Missing required data",
                "details": "Provide patientData and/or riskAssessment"
            }), 400
        img_w, img_h = 1240, 1754
        bg = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(bg)
        try:
            font_title = ImageFont.truetype("Arial.ttf", 48)
            font_h2 = ImageFont.truetype("Arial.ttf", 36)
            font_body = ImageFont.truetype("Arial.ttf", 24)
            font_small = ImageFont.truetype("Arial.ttf", 20)
        except Exception:
            font_title = ImageFont.load_default()
            font_h2 = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_small = ImageFont.load_default()
        header_h = 120
        draw.rectangle([(0, 0), (img_w, header_h)], fill=(179, 0, 0))
        draw.text((40, 30), "CodeCrimsonAI - PPH Risk Report", fill=(255, 255, 255), font=font_title)
        box_margin = 60
        y = header_h + 40
        draw.text((box_margin, y), "Risk Summary", fill=(51, 51, 51), font=font_h2)
        y += 60
        risk_score = risk_assessment.get("riskScore", 0)
        risk_level = risk_assessment.get("riskLevel", "UNKNOWN")
        model_type = risk_assessment.get("modelType", "N/A")
        draw.text((box_margin, y), f"Overall Risk: {risk_score:.1f}%", fill=(220, 38, 38), font=font_h2)
        y += 50
        draw.text((box_margin, y), f"Risk Level: {risk_level}", fill=(80, 80, 80), font=font_body)
        y += 40
        draw.text((box_margin, y), f"Model: {model_type}", fill=(80, 80, 80), font=font_body)
        y += 60
        draw.text((box_margin, y), "Top Risk Factors", fill=(51, 51, 51), font=font_h2)
        y += 50
        top_factors = risk_assessment.get("topRiskFactors", [])
        for idx, f in enumerate(top_factors[:5], start=1):
            name = str(f.get("factor", "-")).strip()
            val = f.get("importance")
            points = f.get("points")
            display = f"{val:.3f}" if isinstance(val, (int, float)) else (str(points) + " pts" if points is not None else "")
            draw.text((box_margin, y), f"{idx}. {name}", fill=(60, 60, 60), font=font_body)
            draw.text((img_w - box_margin - 200, y), display, fill=(179, 0, 0), font=font_body)
            y += 36
        y += 40
        draw.text((box_margin, y), "Patient Details", fill=(51, 51, 51), font=font_h2)
        y += 50
        columns = [
            ("Age (years)", "Systolic Blood Pressure (mmHg)", "Diastolic Blood Pressure (mmHg)"),
            ("Heart Rate (bpm)", "Hemoglobin (g/dL)", "Pre-pregnancy BMI"),
            ("History of PPH", "History of Previous Cesarean", "Placenta Previa"),
            ("Multiple Pregnancy", "Parity", "Gestational Age (weeks)"),
            ("Mode of Delivery", "Estimated Fetal Weight (kg)", "Platelet Count (x10^3/μL)")
        ]
        col_x = [box_margin, img_w//2]
        cur_x = col_x[0]
        cur_y = y
        line_h = 30
        items = []
        for triplet in columns:
            for key in triplet:
                if key in patient_data:
                    items.append((key, patient_data.get(key)))
        half = (len(items) + 1)//2
        for i, (k, v) in enumerate(items):
            cx = col_x[0] if i < half else col_x[1]
            cy = cur_y + (i if i < half else i - half) * line_h
            draw.text((cx, cy), f"{k}: ", fill=(90, 90, 90), font=font_small)
            draw.text((cx + 370, cy), str(v), fill=(40, 40, 40), font=font_small)
        draw.text((box_margin, img_h - 80), "Generated by HemoSync AI - For clinical decision support only", fill=(120,120,120), font=font_small)
        pdf_io = io.BytesIO()
        bg.save(pdf_io, format="PDF", resolution=200.0)
        pdf_io.seek(0)
        from flask import send_file
        return send_file(
            pdf_io,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="pph_risk_report.pdf"
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

@app.route("/api/getNearestBanks", methods=["POST", "OPTIONS"])
def get_nearest_banks():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        if not BLOOD_BANKS_LOADED or blood_banks_df is None:
            return jsonify({
                "error": "Blood bank database not available",
                "success": False
            }), 503
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        lat = data.get("lat")
        lon = data.get("lon")
        blood_type = data.get("bloodType")
        if blood_type == 'N/A':
            blood_type = None
        radius = data.get("radius", 100)
        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude required"}), 400
        patient_loc = (float(lat), float(lon))
        results = []
        for _, row in blood_banks_df.iterrows():
            try:
                bank_loc = (row['lat'], row['lon'])
                dist = haversine(patient_loc, bank_loc, unit=Unit.KILOMETERS)
                if dist <= radius:
                    if blood_type is None or blood_type == "":
                        blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
                        total_units = sum([int(row[bt]) for bt in blood_types if bt in row])
                        if total_units > 0:
                            inventory = {bt: int(row[bt]) for bt in blood_types if bt in row and int(row[bt]) > 0}
                            results.append({
                                'name': row['name'],
                                'distance_km': round(dist, 2),
                                'units_available': total_units,
                                'inventory': inventory,
                                'expiry_dates': row['expiry_dates'],
                                'lat': float(row['lat']),
                                'lon': float(row['lon']),
                                'blood_type': 'All Types',
                                'phone': row.get('phone', '')
                            })
                    elif row[blood_type] > 0:
                        results.append({
                            'name': row['name'],
                            'distance_km': round(dist, 2),
                            'units_available': int(row[blood_type]),
                            'expiry_dates': row['expiry_dates'],
                            'lat': float(row['lat']),
                            'lon': float(row['lon']),
                            'blood_type': blood_type,
                            'phone': row.get('phone', '')
                        })
            except Exception as e:
                continue
        results = sorted(results, key=lambda x: x['distance_km'])[:10]
        return jsonify({
            "success": True,
            "patient_location": {"lat": lat, "lon": lon},
            "blood_type": blood_type if blood_type else "All Types",
            "radius_km": radius,
            "banks_found": len(results),
            "banks": results
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/api/patients", methods=["GET", "OPTIONS"])
def get_patients():
    """Return list of patients and their risk levels."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    try:
        if not PATIENTS_LOADED or patients_df is None:
            return jsonify({
                "error": "Patient database not available",
                "success": False
            }), 503

        patients = []
        for _, row in patients_df.iterrows():
            # Calculate risk score if not provided
            risk_score = row.get('risk_score', None)
            if risk_score is None or pd.isna(risk_score):
                risk_data = {
                    "Age (years)": row.get('age', 0),
                    "Systolic Blood Pressure (mmHg)": row.get('systolic_bp', 0),
                    "Diastolic Blood Pressure (mmHg)": row.get('diastolic_bp', 0),
                    "Heart Rate (bpm)": row.get('heart_rate', 0),
                    "Hemoglobin (g/dL)": row.get('hemoglobin', 0),
                    "Pre-pregnancy BMI": row.get('bmi', 0)
                }
                risk_assessment = calculate_pph_risk(risk_data)
                risk_score = risk_assessment['riskScore']
            
            # Safely convert values to JSON-serializable types
            patient_id = row.get('patient_id', f'MH2024-{len(patients)+1:03d}')
            patient_name = row.get('name', f'Patient {len(patients)+1}')
            patient_age = row.get('age', None)
            patient_edd = row.get('expected_delivery_date', 'N/A')
            patient_status = row.get('status', 'Monitoring')
            patient_trend = row.get('trend', 'stable')
            
            # Convert pandas types to native Python types
            if pd.isna(patient_age):
                patient_age = 'N/A'
            elif isinstance(patient_age, (int, float)):
                patient_age = int(patient_age) if patient_age == int(patient_age) else float(patient_age)
            
            patient_data = {
                "id": str(patient_id),
                "name": str(patient_name),
                "age": patient_age,
                "risk": float(risk_score),
                "edd": str(patient_edd) if not pd.isna(patient_edd) else 'N/A',
                "status": str(patient_status) if not pd.isna(patient_status) else 'Monitoring',
                "trend": str(patient_trend) if not pd.isna(patient_trend) else 'stable'
            }
            patients.append(patient_data)

        response_data = {
            "success": True,
            "patients": patients
        }
        return jsonify(response_data), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/api/alert-blood-bank", methods=["POST"])
def alert_blood_bank():
    """Call the nearest blood bank using Twilio."""
    try:
        data = request.get_json()
        blood_bank_phone = '+917411866860'  # Default blood bank phone number
        patient_name = data.get("patientName", "a patient")
        blood_type = data.get("bloodType", "O+")
        risk_level = data.get("riskLevel", "HIGH")

        if not blood_bank_phone:
            return jsonify({"error": "Blood bank phone number required"}), 400

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml=f"""
            <Response>
                <Say voice="alice" language="en-IN">
                    Hello, this is an urgent alert from CodeCrimson.
                    A patient named {patient_name} is at {risk_level} risk for postpartum hemorrhage.
                    Please reserve {blood_type} blood units immediately.
                    Thank you.
                </Say>
            </Response>
            """,
            to=blood_bank_phone,
            from_=TWILIO_PHONE_NUMBER
        )

        return jsonify({
            "success": True,
            "callSid": call.sid,
            "message": f"Alert call initiated to {blood_bank_phone}"
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

class AudioMonitor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.source = None
        self.monitoring_thread = None

audio_monitor = AudioMonitor()

def background_monitoring():
    """Background thread function for continuous monitoring."""
    global monitoring_active, latest_alert, alerts_queue
    
    while monitoring_active:
        try:
            if audio_monitor.source:
                # Configure recognizer
                audio_monitor.recognizer.energy_threshold = 300
                audio_monitor.recognizer.dynamic_energy_threshold = True
                audio_monitor.recognizer.pause_threshold = 0.5
                
                # Listen for audio
                audio = audio_monitor.recognizer.listen(
                    audio_monitor.source, 
                    phrase_time_limit=PHRASE_DETECTION_WINDOW,
                    timeout=None
                )
                
                try:
                    text = audio_monitor.recognizer.recognize_google(audio).lower()
                    print(f"\n🎤 Voice detected: {text}")

                    if TARGET_PHRASE in text:
                        print("\n🚨 WARNING! Target phrase detected! Creating call...")
                        
                        # Create alert
                        alert_msg = "Emergency: Target phrase detected in operating theater. Immediate blood supply may be needed."
                        alert_payload = {
                            "message": alert_msg,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "level": "critical"
                        }
                        
                        # Update global alert state
                        latest_alert = alert_payload
                        alerts_queue.put(alert_payload)

                        # Call blood bank
                        try:
                            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                            call = client.calls.create(
                                twiml=f"""
                                <Response>
                                    <Say voice="alice" language="en-IN">
                                        Emergency Alert! Blood loss phrase detected in operating theater.
                                        Immediate blood supply may be needed. Please prepare units immediately.
                                    </Say>
                                </Response>
                                """,
                                to=BLOOD_BANK_PHONE_NUMBER,
                                from_=TWILIO_PHONE_NUMBER
                            )
                            print(f"✅ Alert call initiated: {call.sid}")
                        except Exception as call_error:
                            print(f"❌ Failed to make alert call: {call_error}")
                            
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"❌ Speech recognition error: {e}")
                    
        except Exception as e:
            print(f"🎙️ Listening error: {e}")
            if "Audio source must be entered" in str(e):
                print("Attempting to reconnect microphone...")
                try:
                    mic = sr.Microphone()
                    audio_monitor.source = mic.__enter__()
                    audio_monitor.recognizer.adjust_for_ambient_noise(audio_monitor.source, duration=1)
                except Exception as reconnect_error:
                    print(f"Failed to reconnect microphone: {reconnect_error}")
                    monitoring_active = False
                    break
            continue

@app.route("/api/start-monitoring", methods=["POST"])
def start_audio_monitoring():
    """Start monitoring microphone for target phrase."""
    global monitoring_active
    
    try:
        if monitoring_active:
            return jsonify({
                "success": True,
                "message": "Monitoring already active"
            }), 200

        monitoring_active = True
        
        # Create microphone source and keep it open
        mic = sr.Microphone()
        audio_monitor.source = mic.__enter__()
        
        print("🎙️ Adjusting for ambient noise...")
        # Longer ambient noise adjustment for better calibration
        audio_monitor.recognizer.adjust_for_ambient_noise(audio_monitor.source, duration=2)
        
        # Configure recognizer for better sensitivity
        audio_monitor.recognizer.energy_threshold = 300
        audio_monitor.recognizer.dynamic_energy_threshold = True
        audio_monitor.recognizer.pause_threshold = 0.5
        
        print("🎧 Starting continuous monitoring...")
        
        # Start background monitoring thread
        audio_monitor.monitoring_thread = threading.Thread(
            target=background_monitoring,
            daemon=True
        )
        audio_monitor.monitoring_thread.start()

        return jsonify({
            "success": True,
            "message": "Monitoring started"
        }), 200

    except Exception as e:
        monitoring_active = False
        if audio_monitor.source:
            try:
                mic.__exit__(None, None, None)
                audio_monitor.source = None
            except:
                pass
        print(f"Error starting monitoring: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Error starting monitoring: {str(e)}",
            "success": False
        }), 500

@app.route("/api/stop-monitoring", methods=["POST"])
def stop_audio_monitoring():
    """Stop the microphone monitoring."""
    global monitoring_active
    
    try:
        monitoring_active = False
        
        # Clean up microphone source
        if audio_monitor.source:
            try:
                sr.Microphone().__exit__(None, None, None)
                audio_monitor.source = None
            except:
                pass
        
        # Clear audio queue
        while not audio_queue.empty():
            audio_queue.get()
        
        return jsonify({
            "success": True,
            "message": "Audio monitoring stopped"
        }), 200
    except Exception as e:
        print(f"Error stopping monitoring: {e}")
        return jsonify({
            "error": f"Error stopping monitoring: {str(e)}",
            "success": False
        }), 500

# Add a new route to process the audio queue
@app.route("/api/process-audio", methods=["GET"])
def process_audio():
    """Process any audio in the queue for the target phrase."""
    try:
        if not audio_queue.empty():
            recognizer = sr.Recognizer()
            audio = audio_queue.get()
            
            try:
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")

                if TARGET_PHRASE in text:
                    print("\n🚨 WARNING! Target phrase detected!")
                    # Alert blood bank immediately
                    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                    call = client.calls.create(
                        twiml=f"""
                        <Response>
                            <Say voice="alice" language="en-IN">
                                Emergency Alert! Blood loss phrase detected in operating theater.
                                Immediate blood supply may be needed. Please prepare units immediately.
                            </Say>
                        </Response>
                        """,
                        to=BLOOD_BANK_PHONE_NUMBER,
                        from_=TWILIO_PHONE_NUMBER
                    )
                    return jsonify({
                        "success": True,
                        "warning": True,
                        "message": "Target phrase detected! Blood bank alerted.",
                        "callSid": call.sid
                    }), 200

            except sr.UnknownValueError:
                pass
                
            except sr.RequestError as e:
                print(f"API error: {e}")

        return jsonify({
            "success": True,
            "warning": False
        }), 200

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({
            "error": f"Error processing audio: {str(e)}",
            "success": False
        }), 500

@app.route("/api/check-alerts", methods=["GET"])
def check_alerts():
    """Check for any pending alerts."""
    try:
        if not alerts_queue.empty():
            alert = alerts_queue.get()
            return jsonify({
                "success": True,
                "hasAlert": True,
                "alert": alert
            }), 200
            
        return jsonify({
            "success": True,
            "hasAlert": False
        }), 200
        
    except Exception as e:
        print(f"Error checking alerts: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
# --- Start Server with ngrok ---
if __name__ == "__main__":
    NGROK_AUTH = "2Zlb7k0RLMC57dO272SclbfBfmC_79otQFBLFxkcog1itB7wG"
    try:
        port = 8000
        ngrok.set_auth_token(NGROK_AUTH)
        tunnel = ngrok.connect(addr=port, domain="fadlike-elias-nonnoumenally.ngrok-free.dev")
        public_url = tunnel.public_url
        print("=" * 60)
        print("🚀 HemoSync AI-Powered OCR API Server Started!")
        print(f"📡 Public URL: {public_url}")
        print(f"🔗 OCR Endpoint: {public_url}/api/ocr")
        print(f"🩺 Risk Analysis Endpoint: {public_url}/api/risk-analysis")
        print(f"💚 Health Check: {public_url}/api/health")
        print(f"🤖 ML Model: {'✅ Loaded' if MODEL_LOADED else '⚠️ Using Rule-Based'}")
        print("=" * 60)
        print(f"👉 COPY THESE LINES TO YOUR HTML FRONTEND:")
        print(f"   const OCR_API_URL = '{public_url}/api/ocr';")
        print(f"   const RISK_API_URL = '{public_url}/api/risk-analysis';")
        print("=" * 60)
        app.run(port=port, debug=True, use_reloader=False)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()
        exit(1)