import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_sarimax_optimized():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Prepare Exogenous Variables with LAG to avoid Data Leakage
    # We use physiological signals from the PREVIOUS second to predict CURRENT stress
    df['hr_lag'] = df['heart_rate'].shift(1)
    df['eda_lag'] = df['eda'].shift(1)
    df['temp_lag'] = df['temp'].shift(1)
    
    # Drop first row due to NaN in lags
    df = df.dropna().reset_index(drop=True)
    
    series = df['stress_score'].values
    exog_data = df[['hr_lag', 'eda_lag', 'temp_lag']].values
    
    train_size = int(len(series) * 0.8)
    train_y, test_y = series[:train_size], series[train_size:]
    train_exog, test_exog = exog_data[:train_size], exog_data[train_size:]
    
    history_y = list(train_y)
    history_exog = list(train_exog)
    predictions = []
    
    print("Running SARIMAX(2,1,3)(1,1,1,7) with Lagged Exogenous Factors...")
    start_time = time.time()
    
    for i in range(len(test_y)):
        # Refit window (Last 100 points)
        window_y = history_y[-100:]
        window_exog = history_exog[-100:]
        
        # Current exogenous context (lagged values for the current target)
        current_exog = test_exog[i].reshape(1, -1)
        
        model = SARIMAX(endog=window_y, 
                        exog=window_exog,
                        order=(2, 1, 3), 
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        model_fit = model.fit(disp=False)
        
        # Forecast 1 step ahead using the lagged physiological signals
        yhat = model_fit.forecast(steps=1, exog=current_exog)[0]
        predictions.append(yhat)
        
        history_y.append(test_y[i])
        history_exog.append(test_exog[i])
        
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(test_y)}")

    end_time = time.time()
    
    # 3. Metrics
    mae = mean_absolute_error(test_y, predictions)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2 = r2_score(test_y, predictions)
    mape = mean_absolute_percentage_error(test_y, predictions)

    # 4. Professional Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(test_y, color='black', label='Actual Stress', alpha=0.7)
    plt.plot(predictions, color='#e377c2', linestyle='--', label='SARIMAX Forecast (with Physio Lags)')
    plt.title('SARIMAX: Stress Forecasting with Exogenous Physiological Factors', fontweight='bold')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Stress Score')
    
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nR2: {r2:.4f}'
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('Forecasting_Graphs/07_SARIMAX_Optimized.png', dpi=300)
    
    # 5. Export Metrics
    model_name = "SARIMAX"
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'R2': [r2]
    })
    metrics_df.to_csv(f'metrics_{model_name.lower()}.csv', index=False)
    
    # Update master leaderboard
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, metrics_df], ignore_index=True).to_csv('leaderboard.csv', index=False)

    print(f"SARIMAX Complete. Time: {end_time - start_time:.2f}s, MAE: {mae:.4f}")

if __name__ == "__main__":
    run_sarimax_optimized()
