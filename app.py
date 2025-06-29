import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access variables
metrics_url = os.getenv('METRICS_URL')

app = Flask(__name__)
CORS(app)

# Load your trained XGBoost model (make sure this file exists)
model = joblib.load("xgb_loan_status_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    feature = data.get('features', None)

    if feature is None or len(feature) != 11:
        return jsonify({'error': 'Please provide 11 numeric features.'}), 400

    try:
        features = np.array(feature, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({'error': 'Invalid feature values. Make sure all are numbers.'}), 400

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]
    print(prediction)
    result = "APPROVED" if prediction == 1 else "REJECTED" 
    education = "Non-Graduate"
    if (feature[1] == 1) :
        education = "Graduate"
    
    employed = "No"
    if (feature[2] == 1) :
        employed = "Yes"
    
    prediction_ = "Rejected"
    if (prediction == 1):
        prediction_ = "Approved"

    payload = {
        "no_of_dependents": feature[0],
        "education": education,
        "self_employed": employed,
        "income_annum": feature[3],
        "loan_amount": feature[4], 
        "loan_term": feature[5],
        "cibil_score": feature[6],
        "residential_assets_value": feature[7],
        "commercial_assets_value": feature[8],
        "luxury_assets_value": feature[9],
        "bank_assets_value": feature[10],
        "prediction": prediction_
    }
    print(payload)
    try:
        requests.post(metrics_url, json=payload)
        print("done")
    except Exception as e: 
        print("error: ", e)

    return jsonify({
        "prediction": result,
        "probability": round(float(proba), 4)
    })
    
if __name__ == '__main__':
    app.run(debug=True)
