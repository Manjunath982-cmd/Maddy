from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import sqlite3
from calorie_estimator import CalorieEstimator
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask import session, redirect, url_for, flash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'super-secret-2024'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8n model once at startup
model = YOLO('yolov8n.pt')

# -----------------------
# Database helpers
# -----------------------

DB_PATH = 'food.db'


def init_db():
    """Create the foods table and insert a minimal nutrition dataset if empty."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                calories_per_100g REAL
            )"""
    )

    # Users table
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT
            )"""
    )

    # Meals table (history)
    c.execute(
        """CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT,
                detections TEXT,
                total_calories REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )"""
    )

    # Check if table already populated
    c.execute("SELECT COUNT(*) FROM foods")
    if c.fetchone()[0] == 0:
        sample_data = [
            ("apple", 52),
            ("banana", 89),
            ("orange", 47),
            ("carrot", 41),
            ("hot dog", 290),
            ("pizza", 266),
            ("donut", 452),
            ("cake", 389),
            ("broccoli", 34),
            ("sandwich", 250)
        ]
        c.executemany("INSERT OR IGNORE INTO foods (name, calories_per_100g) VALUES (?, ?)", sample_data)
        conn.commit()

    conn.close()

# Initialise DB and calorie estimator
init_db()
calorie_estimator = CalorieEstimator(DB_PATH)

# -----------------------
# Authentication helpers
# -----------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    # Render a basic upload form
    return render_template('index.html')


def run_detection(image_path: str):
    """Run YOLOv8n on the given image and return basic detection results."""
    results = model(image_path)
    detections = []
    names = model.names

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            detections.append({
                'label': label,
                'confidence': round(conf, 2)
            })
    return detections


@app.route('/upload', methods=['POST'])
def upload():
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save image
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run detection
    detections = run_detection(file_path)

    # Estimate calories
    calorie_info = calorie_estimator.estimate(detections, file_path)

    return jsonify({
        'filename': filename,
        'detections': detections,
        'calories': calorie_info
    })


# Route that handles form POST and renders HTML result

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return render_template('index.html', error="Please choose an image to upload")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    detections = run_detection(path)
    calorie_info = calorie_estimator.estimate(detections, path)

    # Save to history if logged in
    if session.get('user_id'):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO meals(user_id, image_path, detections, total_calories) VALUES (?, ?, ?, ?)",
                  (session['user_id'], filename, str(detections), calorie_info['total_calories']))
        conn.commit()
        conn.close()

    return render_template('index.html', detections=detections, calorie_info=calorie_info, image_url=os.path.join(app.config['UPLOAD_FOLDER'], filename))


# Serve uploaded images

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# -----------------------
# Auth routes
# -----------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            flash('Username and password required')
            return render_template('register.html')

        pw_hash = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, pw_hash))
            conn.commit()
            conn.close()
            flash('Registration successful. Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
            return render_template('register.html')
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
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# -----------------------
# Dashboard
# -----------------------

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image_path, total_calories, created_at FROM meals WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    meals = c.fetchall()
    conn.close()

    # Calculate simple stats
    total_meals = len(meals)
    total_calories = sum(row[1] for row in meals)
    avg_calories = round(total_calories / total_meals, 1) if total_meals else 0

    return render_template('dashboard.html', meals=meals, total_meals=total_meals, total_calories=total_calories, avg_calories=avg_calories)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)