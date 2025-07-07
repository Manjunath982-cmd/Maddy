from flask import Flask, render_template, request, jsonify, flash
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'car-price-prediction-secret-key-2024'

class CarPricePredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
        self.trained = False
        
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data based on realistic patterns
        years = np.random.randint(2003, 2021, n_samples)
        present_prices = np.random.uniform(3, 50, n_samples)
        kms_driven = np.random.randint(10000, 200000, n_samples)
        fuel_types = np.random.choice(['Petrol', 'Diesel', 'CNG'], n_samples, p=[0.6, 0.35, 0.05])
        seller_types = np.random.choice(['Individual', 'Dealer'], n_samples, p=[0.4, 0.6])
        transmissions = np.random.choice(['Manual', 'Automatic'], n_samples, p=[0.7, 0.3])
        owners = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.4, 0.15, 0.05])
        
        # Calculate selling price with realistic depreciation and factors
        age = 2024 - years
        base_depreciation = present_prices * (0.85 ** age)  # 15% yearly depreciation
        
        # Apply adjustments based on features
        fuel_multiplier = np.where(fuel_types == 'Diesel', 1.1, 
                         np.where(fuel_types == 'Petrol', 1.0, 0.9))
        transmission_multiplier = np.where(transmissions == 'Automatic', 1.05, 1.0)
        seller_multiplier = np.where(seller_types == 'Dealer', 1.02, 0.98)
        owner_multiplier = 1 - (owners * 0.05)  # 5% reduction per previous owner
        km_factor = 1 - (kms_driven / 1000000)  # Small reduction for high km
        
        selling_prices = (base_depreciation * fuel_multiplier * transmission_multiplier * 
                         seller_multiplier * owner_multiplier * km_factor)
        
        # Add some noise
        selling_prices += np.random.normal(0, selling_prices * 0.1)
        selling_prices = np.maximum(selling_prices, 0.5)  # Minimum price
        
        # Create DataFrame
        data = pd.DataFrame({
            'Car_Name': [f'Car_{i}' for i in range(n_samples)],
            'Year': years,
            'Selling_Price': np.round(selling_prices, 2),
            'Present_Price': present_prices,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_types,
            'Seller_Type': seller_types,
            'Transmission': transmissions,
            'Owner': owners
        })
        
        return data
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Create label encoders for categorical variables
        categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
        
        df_encoded = df.copy()
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        return df_encoded
    
    def train_models(self):
        """Train all ML models"""
        # Create or load dataset
        if os.path.exists('car_data.csv'):
            df = pd.read_csv('car_data.csv')
        else:
            df = self.create_sample_dataset()
            df.to_csv('car_data.csv', index=False)
        
        # Prepare data
        df_encoded = self.prepare_data(df)
        
        # Features and target
        X = df_encoded[self.feature_columns]
        y = df_encoded['Selling_Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        self.model_performance = {}
        
        for name, model in models.items():
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.model_performance[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'model': model
            }
            
            self.models[name] = model
        
        self.trained = True
        
        # Save models
        with open('models.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'performance': self.model_performance
            }, f)
    
    def load_models(self):
        """Load pre-trained models"""
        if os.path.exists('models.pkl'):
            with open('models.pkl', 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.label_encoders = data['label_encoders']
                self.scaler = data['scaler']
                self.model_performance = data['performance']
                self.trained = True
                return True
        return False
    
    def predict_price(self, car_details, model_name='Random Forest'):
        """Predict car price using specified model"""
        if not self.trained:
            return None
        
        # Prepare input data
        input_data = pd.DataFrame([car_details])
        input_encoded = self.prepare_data(input_data)
        input_features = input_encoded[self.feature_columns]
        
        # Get model
        model = self.models[model_name]
        
        # Make prediction
        if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            input_scaled = self.scaler.transform(input_features)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_features)[0]
        
        return max(0, prediction)  # Ensure non-negative price

# Initialize predictor
predictor = CarPricePredictor()

@app.route('/')
def index():
    return render_template('car_index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            car_details = {
                'Year': int(request.form['year']),
                'Present_Price': float(request.form['present_price']),
                'Kms_Driven': int(request.form['kms_driven']),
                'Fuel_Type': request.form['fuel_type'],
                'Seller_Type': request.form['seller_type'],
                'Transmission': request.form['transmission'],
                'Owner': int(request.form['owner'])
            }
            
            # Make predictions with all models
            predictions = {}
            if predictor.trained:
                for model_name in predictor.models.keys():
                    pred_price = predictor.predict_price(car_details, model_name)
                    predictions[model_name] = round(pred_price, 2)
            
            return render_template('car_predict.html', 
                                 car_details=car_details, 
                                 predictions=predictions,
                                 model_performance=predictor.model_performance if predictor.trained else {})
        
        except Exception as e:
            flash(f'Error in prediction: {str(e)}')
            return render_template('car_predict.html')
    
    return render_template('car_predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for price prediction"""
    try:
        data = request.json
        prediction = predictor.predict_price(data)
        
        if prediction is not None:
            return jsonify({
                'success': True,
                'predicted_price': round(prediction, 2),
                'currency': 'INR (Lakhs)'
            })
        else:
            return jsonify({'success': False, 'error': 'Model not trained'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analytics')
def analytics():
    """Show model performance analytics"""
    if not predictor.trained:
        flash('Models are being trained. Please wait...')
        return render_template('car_analytics.html')
    
    return render_template('car_analytics.html', 
                         performance=predictor.model_performance)

@app.route('/train_models')
def train_models():
    """Endpoint to trigger model training"""
    try:
        predictor.train_models()
        flash('Models trained successfully!')
    except Exception as e:
        flash(f'Error training models: {str(e)}')
    
    return redirect(url_for('analytics'))

if __name__ == '__main__':
    # Try to load existing models, otherwise train new ones
    if not predictor.load_models():
        print("Training new models...")
        predictor.train_models()
        print("Models trained successfully!")
    
    app.run(debug=True, host='0.0.0.0', port=5001)