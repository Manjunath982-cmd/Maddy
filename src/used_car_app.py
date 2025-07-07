from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import datetime

app = Flask(__name__)

# -----------------------------------------------------------
# Load trained model at startup
# -----------------------------------------------------------

def _load_model():
    """Load the first *.pkl file found inside the models/ folder or the path
    given by the USED_CAR_MODEL_PATH environment variable."""
    model_path = os.getenv('USED_CAR_MODEL_PATH')
    if model_path and os.path.exists(model_path):
        return joblib.load(model_path)

    # Fallback: pick the first pickle file in models/
    models_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'models')
    models_dir = os.path.abspath(models_dir)
    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith('.pkl'):
                return joblib.load(os.path.join(models_dir, fname))
    return None

model = _load_model()
if model is None:
    print("[WARNING] Trained model not found. Predictions will not work until a model is saved in the models/ directory.")

# -----------------------------------------------------------
# Helper for prediction
# -----------------------------------------------------------

def make_prediction(form_data):
    """Convert form data to DataFrame and predict selling price."""
    try:
        year = int(form_data['Year'])
        present_price = float(form_data['Present_Price'])
        kms_driven = int(form_data['Kms_Driven'])
        fuel_type = form_data['Fuel_Type']
        seller_type = form_data['Seller_Type']
        transmission = form_data['Transmission']
        owner = int(form_data['Owner'])

        current_year = datetime.datetime.now().year
        car_age = current_year - year

        df = pd.DataFrame({
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Fuel_Type': [fuel_type],
            'Seller_Type': [seller_type],
            'Transmission': [transmission],
            'Owner': [owner],
            'car_age': [car_age]
        })

        pred = model.predict(df)[0]
        return round(pred, 2)
    except Exception as e:
        return f"Prediction error: {e}"

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return render_template('car_price.html')

# New JSON prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_api():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded.'}), 500

    data = request.get_json(silent=True)
    if not data:
        # Fallback to form-encoded data
        data = request.form.to_dict()
    if not data:
        return jsonify({'success': False, 'error': 'No input received.'}), 400

    try:
        price = make_prediction(data)
        if isinstance(price, str):
            # Error message returned
            return jsonify({'success': False, 'error': price}), 400
        return jsonify({'success': True, 'predicted_price': price})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# -----------------------------------------------------------
# Main launcher
# -----------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)