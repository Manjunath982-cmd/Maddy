from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
from datetime import datetime, timedelta
import json
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from food_detector import FoodDetector
from calorie_estimator import CalorieEstimator
from diet_advisor import DietAdvisor
import uuid

app = Flask(__name__)
app.secret_key = 'food-estimation-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ML models
food_detector = FoodDetector()
calorie_estimator = CalorieEstimator()
diet_advisor = DietAdvisor()

def init_db():
    """Initialize all database tables"""
    conn = sqlite3.connect('food_app.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  age INTEGER,
                  weight REAL,
                  height REAL,
                  gender TEXT,
                  activity_level TEXT,
                  diet_goal TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Food estimates table
    c.execute('''CREATE TABLE IF NOT EXISTS food_estimates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  image_path TEXT,
                  detected_foods TEXT,
                  total_calories REAL,
                  total_volume REAL,
                  is_fresh BOOLEAN DEFAULT TRUE,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Weight tracking table
    c.execute('''CREATE TABLE IF NOT EXISTS weight_tracking
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  weight REAL NOT NULL,
                  recorded_date DATE DEFAULT CURRENT_DATE,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Food database table
    c.execute('''CREATE TABLE IF NOT EXISTS food_database
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  calories_per_100g REAL NOT NULL,
                  protein REAL,
                  carbs REAL,
                  fat REAL,
                  fiber REAL,
                  cuisine_type TEXT,
                  is_indian BOOLEAN DEFAULT FALSE)''')
    
    # Diet suggestions table
    c.execute('''CREATE TABLE IF NOT EXISTS diet_suggestions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  suggestion_type TEXT,
                  suggestion_text TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()
    
    # Populate food database if empty
    populate_food_database()

def populate_food_database():
    """Populate the food database with Indian and International foods"""
    conn = sqlite3.connect('food_app.db')
    c = conn.cursor()
    
    # Check if database is already populated
    c.execute("SELECT COUNT(*) FROM food_database")
    if c.fetchone()[0] > 0:
        conn.close()
        return
    
    # Indian foods
    indian_foods = [
        ("Rice (Basmati)", 345, 7.1, 78, 0.9, 1.3, "Indian", True),
        ("Roti (Wheat)", 297, 11.4, 56, 4.2, 11.2, "Indian", True),
        ("Dal (Lentils)", 116, 9, 20, 0.4, 8, "Indian", True),
        ("Chicken Curry", 165, 25, 5, 6, 1, "Indian", True),
        ("Paneer", 265, 18, 1.2, 20.8, 0, "Indian", True),
        ("Samosa", 308, 6, 23, 22, 3, "Indian", True),
        ("Biryani", 200, 8, 35, 4, 2, "Indian", True),
        ("Dosa", 168, 4, 28, 4, 2, "Indian", True),
        ("Idli", 58, 2, 12, 0.2, 1, "Indian", True),
        ("Chole", 164, 8, 27, 2.6, 12, "Indian", True)
    ]
    
    # International foods
    international_foods = [
        ("Apple", 52, 0.3, 14, 0.2, 2.4, "Fruit", False),
        ("Banana", 89, 1.1, 23, 0.3, 2.6, "Fruit", False),
        ("Orange", 47, 0.9, 12, 0.1, 2.4, "Fruit", False),
        ("Chicken Breast", 165, 31, 0, 3.6, 0, "Protein", False),
        ("Salmon", 208, 20, 0, 13, 0, "Protein", False),
        ("Broccoli", 34, 2.8, 7, 0.4, 2.6, "Vegetable", False),
        ("Pasta", 131, 5, 25, 1.1, 1.8, "Carbs", False),
        ("Bread", 265, 9, 49, 3.2, 2.7, "Carbs", False),
        ("Egg", 155, 13, 1.1, 11, 0, "Protein", False),
        ("Milk", 42, 3.4, 5, 1, 0, "Dairy", False)
    ]
    
    all_foods = indian_foods + international_foods
    
    c.executemany("INSERT INTO food_database (name, calories_per_100g, protein, carbs, fat, fiber, cuisine_type, is_indian) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", all_foods)
    conn.commit()
    conn.close()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form.get('age', type=int)
        weight = request.form.get('weight', type=float)
        height = request.form.get('height', type=float)
        gender = request.form.get('gender')
        activity_level = request.form.get('activity_level')
        diet_goal = request.form.get('diet_goal')
        
        if not username or not email or not password:
            flash('Username, email, and password are required!')
            return render_template('register.html')
        
        password_hash = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            c.execute("""INSERT INTO users (username, email, password_hash, age, weight, height, gender, activity_level, diet_goal) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (username, email, password_hash, age, weight, height, gender, activity_level, diet_goal))
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        
        except sqlite3.IntegrityError:
            flash('Username or email already exists!')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('food_app.db')
        c = conn.cursor()
        c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('food_app.db')
    c = conn.cursor()
    
    # Get recent food estimates
    c.execute("""SELECT detected_foods, total_calories, total_volume, is_fresh, created_at 
                 FROM food_estimates 
                 WHERE user_id = ? 
                 ORDER BY created_at DESC 
                 LIMIT 10""", (session['user_id'],))
    recent_estimates = c.fetchall()
    
    # Get today's total calories
    today = datetime.now().date()
    c.execute("""SELECT SUM(total_calories) FROM food_estimates 
                 WHERE user_id = ? AND DATE(created_at) = ?""", (session['user_id'], today))
    today_calories = c.fetchone()[0] or 0
    
    # Get weight history
    c.execute("""SELECT weight, recorded_date FROM weight_tracking 
                 WHERE user_id = ? 
                 ORDER BY recorded_date DESC 
                 LIMIT 30""", (session['user_id'],))
    weight_history = c.fetchall()
    
    # Get recent diet suggestions
    c.execute("""SELECT suggestion_type, suggestion_text, created_at FROM diet_suggestions 
                 WHERE user_id = ? 
                 ORDER BY created_at DESC 
                 LIMIT 5""", (session['user_id'],))
    suggestions = c.fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                         recent_estimates=recent_estimates,
                         today_calories=today_calories,
                         weight_history=weight_history,
                         suggestions=suggestions)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Detect foods in the image
            detection_results = food_detector.detect_foods(filepath)
            
            if detection_results['error']:
                return jsonify({'error': detection_results['error']})
            
            # Estimate calories and volume
            calorie_results = calorie_estimator.estimate_calories(detection_results)
            
            # Check for food freshness
            freshness_check = food_detector.check_freshness(filepath)
            
            # Save to database
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            c.execute("""INSERT INTO food_estimates (user_id, image_path, detected_foods, total_calories, total_volume, is_fresh) 
                         VALUES (?, ?, ?, ?, ?, ?)""",
                     (session['user_id'], filename, json.dumps(detection_results['foods']), 
                      calorie_results['total_calories'], calorie_results['total_volume'], freshness_check['is_fresh']))
            conn.commit()
            conn.close()
            
            # Generate diet suggestions
            suggestions = diet_advisor.get_suggestions(session['user_id'], calorie_results)
            
            return jsonify({
                'success': True,
                'detected_foods': detection_results['foods'],
                'total_calories': calorie_results['total_calories'],
                'total_volume': calorie_results['total_volume'],
                'is_fresh': freshness_check['is_fresh'],
                'freshness_message': freshness_check['message'],
                'suggestions': suggestions,
                'image_url': url_for('static', filename=f'uploads/{filename}')
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/track_weight', methods=['POST'])
def track_weight():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    weight = request.json.get('weight')
    if not weight:
        return jsonify({'error': 'Weight is required'}), 400
    
    conn = sqlite3.connect('food_app.db')
    c = conn.cursor()
    c.execute("INSERT INTO weight_tracking (user_id, weight) VALUES (?, ?)",
             (session['user_id'], weight))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Weight tracked successfully'})

@app.route('/get_nutrition_report')
def get_nutrition_report():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    # Get nutrition data for the last 7 days
    week_ago = datetime.now() - timedelta(days=7)
    
    conn = sqlite3.connect('food_app.db')
    c = conn.cursor()
    c.execute("""SELECT DATE(created_at) as date, SUM(total_calories) as calories
                 FROM food_estimates 
                 WHERE user_id = ? AND created_at >= ?
                 GROUP BY DATE(created_at)
                 ORDER BY date""", (session['user_id'], week_ago))
    
    weekly_data = c.fetchall()
    conn.close()
    
    return jsonify({'weekly_calories': weekly_data})

@app.route('/food_database')
def food_database():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    search = request.args.get('search', '')
    cuisine = request.args.get('cuisine', '')
    
    conn = sqlite3.connect('food_app.db')
    c = conn.cursor()
    
    query = "SELECT name, calories_per_100g, protein, carbs, fat, cuisine_type FROM food_database WHERE 1=1"
    params = []
    
    if search:
        query += " AND name LIKE ?"
        params.append(f"%{search}%")
    
    if cuisine:
        query += " AND cuisine_type = ?"
        params.append(cuisine)
    
    query += " ORDER BY name"
    
    c.execute(query, params)
    foods = c.fetchall()
    conn.close()
    
    return render_template('food_database.html', foods=foods, search=search, cuisine=cuisine)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)