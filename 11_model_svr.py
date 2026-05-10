import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_svr_forecast():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Feature Engineering: Lagged Features
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
    
    # SVR is sensitive to scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    print("Training Support Vector Regression (SVR)...")
    start_time = time.time()
    
    # Using RBF kernel for non-linear temporal patterns
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train_scaled, y_train_scaled)
    
    # 4. Forecast
    predictions_scaled = model.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    end_time = time.time()
    
    # 5. Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    # 6. Professional Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='black', label='Observed Stress', alpha=0.7)
    plt.plot(predictions, color='#bcbd22', linestyle='--', label='SVR Forecast')
    plt.title('Machine Learning: Support Vector Regression (SVR) Stress Forecasting', fontweight='bold')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Stress Score')
    
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nR2: {r2:.4f}'
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('Forecasting_Graphs/11_SVR_Forecast.png', dpi=300)
    plt.close()
    
    # 7. Export Metrics
    model_name = "SVR"
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'R2': [r2]
    })
    metrics_df.to_csv(f'metrics_svr.csv', index=False)
    
    # Update master leaderboard
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, metrics_df], ignore_index=True).to_csv('leaderboard.csv', index=False)

    print(f"SVR Complete. Time: {end_time - start_time:.2f}s, MAE: {mae:.4f}")

if __name__ == "__main__":
    run_svr_forecast()
