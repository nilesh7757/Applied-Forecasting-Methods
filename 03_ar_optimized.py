import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_ar_optimized():
    df = pd.read_csv('s2_proper_continuous.csv')
    series = df['stress_score'].values
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # AIC Optimized AR(2)
    history = list(train)
    predictions = []
    
    print("Running AR(2) Rolling Forecast...")
    for i in range(len(test)):
        model = ARIMA(history[-50:], order=(2, 1, 0))
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(test[i])
    
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    r2 = r2_score(test, predictions)

    plt.figure(figsize=(14, 6))
    plt.plot(test, color='black', label='Observed', alpha=0.7)
    plt.plot(predictions, color='red', linestyle='--', label='AR(2) Forecast')
    plt.title('Optimal AutoRegressive (AR) Model: Stress Forecasting', fontweight='bold')
    plt.legend()
    plt.savefig('Forecasting_Graphs/03_AR_Optimized.png', dpi=300)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Model': ['AR(2)'], 
        'MAE': [mae], 
        'RMSE': [rmse], 
        'MAPE': [mape], 
        'R2': [r2]
    })
    metrics_df.to_csv('leaderboard.csv', index=False)
    print(f"AR(2) Complete. MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    run_ar_optimized()

