THIS SHOULD BE A LINTER ERRORfrom flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from datetime import datetime
import pickle
import numpy as np
from food_estimator import FoodEstimator

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS food_estimates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  food_name TEXT NOT NULL,
                  estimated_calories REAL,
                  estimated_weight REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()

# Initialize food estimator
food_estimator = FoodEstimator()

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
        
        if not username or not email or not password:
            flash('All fields are required!')
            return render_template('register.html')
        
        # Hash the password
        password_hash = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                     (username, email, password_hash))
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
        
        conn = sqlite3.connect('users.db')
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
    
    # Get recent food estimates for this user
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""SELECT food_name, estimated_calories, estimated_weight, created_at 
                 FROM food_estimates 
                 WHERE user_id = ? 
                 ORDER BY created_at DESC 
                 LIMIT 10""", (session['user_id'],))
    recent_estimates = c.fetchall()
    conn.close()
    
    return render_template('dashboard.html', recent_estimates=recent_estimates)

@app.route('/estimate', methods=['POST'])
def estimate_food():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    food_name = request.json.get('food_name', '').strip()
    if not food_name:
        return jsonify({'error': 'Food name is required'}), 400
    
    # Get estimation from ML model
    estimation = food_estimator.estimate(food_name)
    
    # Save to database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""INSERT INTO food_estimates (user_id, food_name, estimated_calories, estimated_weight) 
                 VALUES (?, ?, ?, ?)""",
             (session['user_id'], food_name, estimation['calories'], estimation['weight']))
    conn.commit()
    conn.close()
    
    return jsonify(estimation)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)