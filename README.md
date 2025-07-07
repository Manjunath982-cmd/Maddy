# 🚗 Used Car Price Prediction Web Application

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.1.3-purple.svg)](https://getbootstrap.com/)

## 📋 Project Overview

This project is a comprehensive web application that predicts used car prices using machine learning algorithms. The application provides real-time price estimation based on various car attributes and features a modern, user-friendly interface built with Flask and Bootstrap.

### 🎯 Key Features

- **🤖 5 Machine Learning Models**: Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting
- **👤 User Authentication**: Complete registration and login system
- **📊 Interactive Analytics**: Model performance visualization with charts
- **💾 Prediction History**: Save and track user predictions
- **📱 Responsive Design**: Modern UI with Bootstrap 5
- **⚡ Real-time Predictions**: Instant price estimates from multiple models

## 🛠️ Tech Stack

### Backend
- **Python 3.10** - Programming language
- **Flask 2.3.3** - Web framework
- **SQLite** - Database for user data and predictions
- **scikit-learn 1.3.0** - Machine learning models
- **pandas 2.0.3** - Data manipulation
- **NumPy 1.24.3** - Numerical computing

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Bootstrap 5.1.3** - Responsive design framework
- **Font Awesome 6.0** - Icons
- **Chart.js** - Data visualization
- **JavaScript ES6** - Interactive functionality

### Machine Learning Models
1. **Linear Regression** - Fast, interpretable baseline model
2. **Ridge Regression** - L2 regularized linear model
3. **Lasso Regression** - L1 regularized with feature selection
4. **Random Forest** - Ensemble method (best performing)
5. **Gradient Boosting** - Sequential ensemble learning

## 📊 Dataset Information

- **Size**: 1,000+ synthetic car records
- **Features**: 8 input features + target variable
- **Brands**: 15+ popular Indian and international brands
- **Years**: 2003-2020 manufacturing years
- **Price Range**: ₹0.5L - ₹80L realistic pricing

### Input Features
1. **Year** - Manufacturing year (2003-2020)
2. **Present_Price** - Current market price when new (₹ Lakhs)
3. **Kms_Driven** - Total kilometers driven
4. **Fuel_Type** - Petrol, Diesel, or CNG
5. **Seller_Type** - Individual or Dealer
6. **Transmission** - Manual or Automatic
7. **Owner** - Number of previous owners (0-3+)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd car-price-prediction
```

2. **Install dependencies**
```bash
pip install -r car_requirements.txt
```

3. **Generate dataset (if needed)**
```bash
python3 create_simple_dataset.py
```

4. **Run the application**
```bash
python3 car_price_app.py
```

5. **Access the application**
Open your browser and navigate to: `http://localhost:5001`

### Alternative Setup with Virtual Environment
```bash
python3 -m venv car_env
source car_env/bin/activate  # On Windows: car_env\Scripts\activate
pip install -r car_requirements.txt
python3 car_price_app.py
```

## 📁 Project Structure

```
car-price-prediction/
├── car_price_app.py           # Main Flask application
├── create_dataset.py          # Advanced dataset generator (requires pandas)
├── create_simple_dataset.py   # Simple dataset generator (built-in libs only)
├── car_requirements.txt       # Python dependencies
├── car_data.csv              # Generated dataset
├── models.pkl                # Trained ML models (auto-generated)
├── car_price_app.db          # SQLite database (auto-generated)
├── templates/
│   ├── car_base.html         # Base template with navigation
│   ├── car_index.html        # Landing page
│   ├── car_login.html        # User login page
│   ├── car_register.html     # User registration page
│   ├── car_predict.html      # Main prediction interface
│   ├── car_analytics.html    # Model performance analytics
│   └── car_profile.html      # User profile and history
└── README.md                 # This file
```

## 🎮 Usage Guide

### 1. **Home Page**
- Overview of features and capabilities
- Quick access to prediction and analytics
- Information about ML models used

### 2. **User Registration/Login**
- Create account to save prediction history
- Secure authentication with password hashing
- Guest access available for immediate predictions

### 3. **Price Prediction**
- Fill in car details form
- Get instant predictions from 5 ML models
- View recommended price range
- Save predictions (logged-in users)

### 4. **Analytics Dashboard**
- Model performance comparison
- R² scores, MAE, and RMSE metrics
- Interactive charts and visualizations
- Model explanations and recommendations

### 5. **User Profile**
- View prediction history
- Account statistics
- Export prediction data

## 🧠 Machine Learning Details

### Model Performance (Typical Results)
| Model | R² Score | MAE (₹L) | RMSE (₹L) | Speed |
|-------|----------|----------|-----------|-------|
| Random Forest | 0.95+ | 0.8-1.2 | 1.0-1.5 | Medium |
| Gradient Boosting | 0.94+ | 0.9-1.3 | 1.1-1.6 | Medium |
| Ridge Regression | 0.88+ | 1.2-1.8 | 1.5-2.2 | Fast |
| Lasso Regression | 0.87+ | 1.3-1.9 | 1.6-2.3 | Fast |
| Linear Regression | 0.85+ | 1.4-2.0 | 1.7-2.5 | Fast |

### Feature Engineering
- **Depreciation Modeling**: Realistic 12-15% annual depreciation
- **Brand Tier Adjustments**: Premium brand multipliers
- **Usage Impact**: Kilometer-based price reduction
- **Market Factors**: Fuel type, transmission, seller type adjustments

## 🔧 Configuration

### Application Settings
- **Debug Mode**: Enabled by default (disable for production)
- **Port**: 5001 (configurable)
- **Database**: SQLite (easily replaceable)
- **Secret Key**: Change for production deployment

### Model Parameters
- **Random Forest**: 100 estimators, random_state=42
- **Gradient Boosting**: 100 estimators, random_state=42
- **Regularization**: Ridge α=1.0, Lasso α=0.1

## 🔍 API Endpoints

### Main Routes
- `GET /` - Home page
- `GET,POST /register` - User registration
- `GET,POST /login` - User login
- `GET /logout` - User logout
- `GET,POST /predict` - Price prediction interface
- `GET /analytics` - Model performance analytics
- `GET /profile` - User profile and history

### API Routes
- `POST /api/predict` - JSON API for price prediction
- `GET /train_models` - Trigger model retraining

## 🎨 UI/UX Features

### Design Elements
- **Gradient Backgrounds**: Modern glassmorphism design
- **Responsive Layout**: Mobile-first Bootstrap grid
- **Interactive Forms**: Real-time validation and hints
- **Charts & Visualizations**: Chart.js integration
- **Icons**: Font Awesome 6.0 icons throughout

### User Experience
- **Loading States**: Visual feedback during predictions
- **Form Validation**: Client and server-side validation
- **Error Handling**: User-friendly error messages
- **Accessibility**: ARIA labels and keyboard navigation

## 🚀 Deployment

### Development
```bash
python3 car_price_app.py
```

### Production (with Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5001 car_price_app:app
```

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

## 🔮 Future Enhancements

### Planned Features
- [ ] Location-based pricing adjustments
- [ ] Image-based condition analysis
- [ ] Price trend analysis
- [ ] Email notifications for price alerts
- [ ] API rate limiting and authentication
- [ ] Advanced filtering and search
- [ ] Car comparison tool
- [ ] Market insights dashboard

### Technical Improvements
- [ ] Database migration to PostgreSQL
- [ ] Redis caching for model predictions
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Automated testing suite
- [ ] Performance monitoring
- [ ] Security hardening

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** community for excellent ML libraries
- **Flask** team for the lightweight web framework
- **Bootstrap** for responsive design components
- **Chart.js** for beautiful data visualizations
- Used car market data insights from various automotive platforms

## 📞 Support

For support, questions, or suggestions:
- Create an issue in the repository
- Contact: [your-email@example.com]
- Documentation: [project-docs-url]

---

**Built with ❤️ using Python, Flask, and Machine Learning**