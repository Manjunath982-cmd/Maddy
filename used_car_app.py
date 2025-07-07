from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join('models', 'best_model.pkl')

app = Flask(__name__)

# Load model once
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def model_ready():
    return model is not None


@app.route('/', methods=['GET', 'POST'])
def index():
    if not model_ready():
        return "Model file not found. Train the model by running train_model.py first.", 500

    if request.method == 'POST':
        # Extract form values
        year = int(request.form['Year'])
        present_price = float(request.form['Present_Price'])
        kms_driven = int(request.form['Kms_Driven'])
        fuel_type = request.form['Fuel_Type']
        transmission = request.form['Transmission']
        seller_type = request.form['Seller_Type']
        owner = int(request.form['Owner'])

        # Build single-sample dataframe in same order as training
        import pandas as pd
        sample = pd.DataFrame({
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission],
            'Seller_Type': [seller_type],
            'Owner': [owner]
        })

        pred = model.predict(sample)[0]
        pred = round(pred, 2)
        return render_template('car_index.html', prediction=pred, form=request.form)

    return render_template('car_index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)