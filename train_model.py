import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os

DATA_PATH = 'car_data.csv'  # <-- place your Kaggle dataset here
MODEL_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_columns.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)



def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Rename columns to remove spaces
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    return df


def build_preprocessor(df):
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    num_cols = [c for c in df.columns if df[c].dtype != 'object' and c != 'Selling_Price']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    return preprocessor, cat_cols, num_cols


def train_models(X_train, y_train, preprocessor):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }

    trained = {}
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        trained[name] = pipe
    return trained


def evaluate_models(models, X_test, y_test):
    scores = {}
    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        scores[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    return scores


if __name__ == '__main__':
    df = load_data()
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])

    preprocessor, cat_cols, num_cols = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_models = train_models(X_train, y_train, preprocessor)
    scores = evaluate_models(trained_models, X_test, y_test)

    # Pick best by R2
    best_name = max(scores, key=lambda k: scores[k]['R2'])
    best_model = trained_models[best_name]

    print('Model scores:')
    for n, s in scores.items():
        print(f"{n}: R2={s['R2']:.3f}, MAE={s['MAE']:.2f}, RMSE={s['RMSE']:.2f}")
    print(f"\nSelected best model: {best_name}")

    # Save model
    joblib.dump(best_model, BEST_MODEL_PATH)
    # Save feature names after preprocessing (need for predicting later)
    cols = list(X.columns)
    joblib.dump(cols, FEATURES_PATH)
    print(f"Saved best model to {BEST_MODEL_PATH}")