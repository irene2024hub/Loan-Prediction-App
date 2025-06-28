import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

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

    result = "APPROVED" if prediction == 1 else "REJECTED"

    return jsonify({
        "prediction": result,
        "probability": round(float(proba), 4)
    })
    print(feature)
    
if __name__ == '__main__':
    app.run(debug=True)
