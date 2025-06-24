import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import os

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# --- Load trained models ---
models_path = r"D:\DA and DS Projects\Uber_Trip_Analysis\models"  # raw string to avoid \U escape issues
xgb_model = joblib.load(os.path.join(models_path, "xgb_model.pkl"))
rf_model = joblib.load(os.path.join(models_path, "rf_model.pkl"))
gbr_model = joblib.load(os.path.join(models_path, "gbr_model.pkl"))
weights = joblib.load(os.path.join(models_path, "ensemble_weights.pkl"))
models = {"XGB": xgb_model, "RF": rf_model, "GBR": gbr_model}

# --- App ---
st.set_page_config(layout="wide")
st.title("🚕 Uber Hourly Trips Forecast")

@st.cache_data
def load_data():
    file_path = 'uber-raw-data-janjune-15.csv'
    df = pd.read_csv(file_path, parse_dates=['Pickup_date'])
    df.set_index('Pickup_date', inplace=True)
    hourly = df.resample('h').size().to_frame('trip_count')
    hourly['hour'] = hourly.index.hour
    hourly['dayofweek'] = hourly.index.dayofweek
    hourly['month'] = hourly.index.month
    hourly['is_weekend'] = (hourly['dayofweek'] >= 5).astype(int)
    hourly['roll_mean_24h'] = hourly['trip_count'].rolling(24).mean()
    hourly['roll_std_24h'] = hourly['trip_count'].rolling(24).std()
    for lag in range(1, 25):
        hourly[f'lag_{lag}'] = hourly['trip_count'].shift(lag)
    hourly.dropna(inplace=True)
    return hourly

def evaluate_model(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mape, rmse, r2

def forecast_hours(history_df, models, weights, N=24):
    df = history_df[['trip_count']].copy()
    forecasts = []
    for _ in range(N):
        t = df.index[-1] + pd.Timedelta(hours=1)
        feat = {
            'hour': t.hour,
            'dayofweek': t.dayofweek,
            'month': t.month,
            'is_weekend': int(t.dayofweek >= 5),
            'roll_mean_24h': df['trip_count'][-24:].mean(),
            'roll_std_24h': df['trip_count'][-24:].std()
        }
        for lag in range(1, 25):
            feat[f'lag_{lag}'] = df['trip_count'].iloc[-lag]
        Xn = pd.DataFrame(feat, index=[t])
        model_preds = {n: m.predict(Xn)[0] for n, m in models.items()}
        ens = sum(model_preds[n] * weights[n] for n in model_preds)
        df.loc[t, 'trip_count'] = ens
        forecasts.append((t, ens))
    return pd.DataFrame(forecasts, columns=['datetime', 'forecast']).set_index('datetime')

# --- Load and Prepare Data ---
hourly = load_data()
cutoff = int(len(hourly) * 0.8)
train, test = hourly.iloc[:cutoff], hourly.iloc[cutoff:]
X_test, y_test = test.drop('trip_count', axis=1), test['trip_count']

# --- Predictions ---
with st.spinner("Generating predictions..."):
    preds = {name: model.predict(X_test) for name, model in models.items()}
    ensemble_preds = sum(preds[name] * weights[name] for name in preds)

# --- Metrics ---
st.subheader("📊 Test Set Metrics")
for name, pred in preds.items():
    mape, rmse, r2 = evaluate_model(y_test, pred)
    st.write(f"**{name}** - MAPE: {mape:.2%}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
mape, rmse, r2 = evaluate_model(y_test, ensemble_preds)
st.write(f"**Ensemble** - MAPE: {mape:.2%}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

# --- Forecast ---
future_df = forecast_hours(hourly, models, weights, N=24)

# --- Plotly Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=y_test, mode='lines', name='Actual'))
for name, p in preds.items():
    fig.add_trace(go.Scatter(x=test.index, y=p, mode='lines', name=name))
fig.add_trace(go.Scatter(x=test.index, y=ensemble_preds,
                         mode='lines', name='Ensemble', line=dict(width=4)))
fig.add_trace(go.Scatter(x=future_df.index, y=future_df['forecast'],
                         mode='lines', name='24h Forecast', line=dict(dash='dash')))
fig.update_layout(
    title="Uber Hourly Trips: Actual vs Predictions & 24‑Hour Forecast",
    xaxis_title="Datetime",
    yaxis_title="Trip Count",
    template="plotly_white",
    height=600
)
st.plotly_chart(fig, use_container_width=True)


# Add explanation
st.markdown("""
    **Ensemble Model Explanation:**
    The ensemble combines the predictions from 3 models: XGB, Random Forest, and Gradient Boosting.
    Each model has a different weighting based on its performance, and the final prediction is a weighted average of these models.
    The forecast is generated based on recent trip data, including rolling statistics (mean, standard deviation) and lag features.
""")

st.success("App ready!")

