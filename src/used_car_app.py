from flask import Flask, render_template, request
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

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        if model is None:
            prediction = "Model not loaded. Train and save a model in the models/ folder first."
        else:
            prediction = make_prediction(request.form)
    return render_template('car_price.html', prediction=prediction)

# -----------------------------------------------------------
# Main launcher
# -----------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)