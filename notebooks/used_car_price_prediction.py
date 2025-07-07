# %% [markdown]
# # Used Car Price Prediction
# ### Data Processing, Exploration, and Model Development
#
# This notebook (Python script with Jupytext cell markers) walks through the full machine-learning pipeline for predicting the selling price of a used car.
# Feel free to execute each cell sequentially, tweak parameters, and add further analysis.

# %% [markdown]
# ## 0. Imports & Setup

# %%
# Standard libraries
import os, warnings, joblib

# Data manipulation & visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine-learning utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# %% [markdown]
# ## 1. Load Dataset

# %%
# Update the path to your local copy of the CSV file as necessary
DATA_PATH = '../data/raw/car data.csv'  # <-- change me if needed

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please update the path.")

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %%
df.info()

# Visualize numeric feature distributions
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.show()

# %% [markdown]
# ## 3. Feature Engineering & Pre-Processing

# %%
# Example: create car age from 'Year' column
CURRENT_YEAR = 2024
df['car_age'] = CURRENT_YEAR - df['Year']

# Drop columns that won't be used directly
if 'Car_Name' in df.columns:
    df = df.drop(['Car_Name'], axis=1)

df = df.drop(['Year'], axis=1)

# Define feature matrix X and target y
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Column categorization
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

# Preprocess: one-hot encode categorical features, passthrough numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# %% [markdown]
# ## 4. Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## 5. Model Training & Evaluation

# %%
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.001),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[('prep', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Pipeline': pipeline})

results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
results_df[['Model', 'MAE', 'RMSE', 'R2']]

# %% [markdown]
# ## 6. Persist Best Model

# %%
best_row = results_df.loc[0]
best_model_name = best_row['Model']
best_pipeline = best_row['Pipeline']

MODEL_DIR = '../models'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"{best_model_name}_regressor.pkl")
joblib.dump(best_pipeline, MODEL_PATH)
print(f"Best model ({best_model_name}) saved to -> {MODEL_PATH}")

# %% [markdown]
# ## 7. Next Steps
# * Perform hyperparameter tuning (e.g., GridSearchCV) for further improvements.
# * Analyze feature importance (e.g., `best_pipeline.named_steps['model'].feature_importances_` for tree-based models).
# * Integrate the saved model into the Flask web application for real-time inference.