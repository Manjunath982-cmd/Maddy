from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
import os, sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps

import pandas as pd
from train_model import build_preprocessor, train_models, evaluate_models  # reuse helpers

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
DB_PATH = 'car_users.db'
UPLOAD_DIR = 'data_uploads'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'used-car-secret-2024'

# ---------------------
# Database helpers
# ---------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT
            )""")
    conn.commit()
    conn.close()

init_db()

# ---------------------
# Auth helpers
# ---------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please login first')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ---------------------
# Load model (if exists)
# ---------------------

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

# ---------------------
# Routes
# ---------------------

@app.route('/')
@login_required
def home():
    return redirect(url_for('predict'))


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if model is None:
        flash('Model not trained yet. Upload data on /train to create model')
        return render_template('car_index.html', prediction=None)

    if request.method == 'POST':
        # Extract form values
        form = request.form
        sample = pd.DataFrame({
            'Year': [int(form['Year'])],
            'Present_Price': [float(form['Present_Price'])],
            'Kms_Driven': [int(form['Kms_Driven'])],
            'Fuel_Type': [form['Fuel_Type']],
            'Transmission': [form['Transmission']],
            'Seller_Type': [form['Seller_Type']],
            'Owner': [int(form['Owner'])]
        })
        pred = model.predict(sample)[0]
        return render_template('car_index.html', prediction=round(pred,2), form=form)

    return render_template('car_index.html', prediction=None)


# -------- Auth --------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            flash('Both fields required')
            return render_template('register.html')
        pw_hash = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users(username, password_hash) VALUES(?, ?)", (username, pw_hash))
            conn.commit()
            conn.close()
            flash('Registration successful. Login now')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already taken')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[1], password):
            session['user_id'] = row[0]
            session['username'] = username
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Logged out')
    return redirect(url_for('login'))

# -------- Training --------

@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        file = request.files['dataset']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_DIR, filename)
        file.save(path)
        flash('Dataset uploaded. Training model, please wait...')
        # Train model synchronously (for simplicity)
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().replace(' ', '_') for c in df.columns]
            y = df['Selling_Price']
            X = df.drop(columns=['Selling_Price'])
            preprocessor, _, _ = build_preprocessor(df)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            trained = train_models(X_train, y_train, preprocessor)
            scores = evaluate_models(trained, X_test, y_test)
            best_name = max(scores, key=lambda k: scores[k]['R2'])
            best_model = trained[best_name]
            joblib.dump(best_model, MODEL_PATH)
            load_model()
            flash(f'Model trained successfully using {best_name}. R2: {scores[best_name]["R2"]:.3f}')
        except Exception as e:
            flash(f'Training failed: {e}')
        return redirect(url_for('train'))

    return render_template('train.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)