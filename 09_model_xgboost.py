import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_xgboost_forecast():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Feature Engineering: Lagged Features (Avoid Leakage)
    df['stress_lag1'] = df['stress_score'].shift(1)
    df['stress_lag2'] = df['stress_score'].shift(2)
    df['hr_lag1'] = df['heart_rate'].shift(1)
    df['eda_lag1'] = df['eda'].shift(1)
    df['temp_lag1'] = df['temp'].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    features = ['stress_lag1', 'stress_lag2', 'hr_lag1', 'eda_lag1', 'temp_lag1']
    X = df[features].values
    y = df['stress_score'].values
    
    # 3. Train/Test Split
    train_size = int(len(y) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Training XGBoost Regressor...")
    start_time = time.time()
    
    # XGBoost Hyperparameters for Time Series
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50 # Moved here from fit()
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    
    # 4. Forecast
    predictions = model.predict(X_test)
    
    end_time = time.time()
    
    # 5. Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    # 6. Professional Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='black', label='Observed Stress', alpha=0.7)
    plt.plot(predictions, color='#ff7f0e', linestyle='--', label='XGBoost Forecast')
    plt.title('Machine Learning: XGBoost Gradient Boosting Stress Forecasting', fontweight='bold')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Stress Score')
    
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nR2: {r2:.4f}'
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('Forecasting_Graphs/09_XGBoost_Forecast.png', dpi=300)
    plt.close()
    
    # 7. Export Metrics
    model_name = "XGBoost"
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'R2': [r2]
    })
    metrics_df.to_csv(f'metrics_xgboost.csv', index=False)
    
    # Update master leaderboard
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, metrics_df], ignore_index=True).to_csv('leaderboard.csv', index=False)

    print(f"XGBoost Complete. Time: {end_time - start_time:.2f}s, MAE: {mae:.4f}")

if __name__ == "__main__":
    run_xgboost_forecast()
