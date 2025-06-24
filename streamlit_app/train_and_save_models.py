import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

# --- Load and process data ---
file_path = 'uber-raw-data-janjune-15.csv'
df = pd.read_csv(file_path, parse_dates=['Pickup_date'])
df.set_index('Pickup_date', inplace=True)
hourly = df.resample('h').size().to_frame('trip_count')
hourly['hour']       = hourly.index.hour
hourly['dayofweek']  = hourly.index.dayofweek
hourly['month']      = hourly.index.month
hourly['is_weekend'] = (hourly['dayofweek'] >= 5).astype(int)
hourly['roll_mean_24h'] = hourly['trip_count'].rolling(24).mean()
hourly['roll_std_24h']  = hourly['trip_count'].rolling(24).std()
for lag in range(1, 25):
    hourly[f'lag_{lag}'] = hourly['trip_count'].shift(lag)
hourly.dropna(inplace=True)

# --- Split ---
cutoff = int(len(hourly) * 0.8)
train = hourly.iloc[:cutoff]
X_train, y_train = train.drop('trip_count', axis=1), train['trip_count']

# --- Model configs ---
tscv = TimeSeriesSplit(n_splits=5)
models_cfg = {
    'XGB': XGBRegressor(objective='reg:squarederror', random_state=42),
    'RF':  RandomForestRegressor(random_state=42),
    'GBR': GradientBoostingRegressor(random_state=42)
}
params_dist = {
    'XGB': {
        'n_estimators': [200],
        'max_depth':    [6],
        'learning_rate':[0.1],
        'subsample':    [0.8],
        'colsample_bytree':[1.0]
    },
    'RF': {
        'n_estimators': [100],
        'max_depth':    [20],
        'min_samples_split':[5],
        'min_samples_leaf':[2]
    },
    'GBR': {
        'n_estimators': [200],
        'learning_rate':[0.1],
        'max_depth':    [5],
        'min_samples_split':[2],
        'min_samples_leaf':[1]
    }
}

# --- Train ---
models = {}
for name, model in models_cfg.items():
    search = RandomizedSearchCV(
        model, params_dist[name], n_iter=1, cv=tscv,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    models[name] = search.best_estimator_

# --- Compute ensemble weights ---
val_preds = {name: m.predict(X_train) for name, m in models.items()}
mapes = {name: mean_absolute_percentage_error(y_train, p) for name, p in val_preds.items()}
inv = {name: 1 / mapes[name] for name in mapes}
total = sum(inv.values())
weights = {name: inv[name] / total for name in inv}

# --- Save models ---
os.makedirs('models', exist_ok=True)
for name, model in models.items():
    joblib.dump(model, f'models/{name.lower()}_model.pkl')
joblib.dump(weights, 'models/ensemble_weights.pkl')

print("✅ Models and weights saved to `models/` folder.")


