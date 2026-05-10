import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_arma_optimized():
    df = pd.read_csv('s2_proper_continuous.csv')
    series = df['stress_score'].values
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # AIC Optimized ARMA(2, 3)
    history = list(train)
    predictions = []
    
    print("Running ARMA(2, 3) Rolling Forecast...")
    for i in range(len(test)):
        # ARMA(2, 3) with d=1 is ARIMA(2, 1, 3)
        model = ARIMA(history[-100:], order=(2, 1, 3))
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
    plt.plot(predictions, color='blue', linestyle='--', label='ARMA(2,3) Forecast')
    plt.title('Optimal AutoRegressive Moving Average (ARMA) Model', fontweight='bold')
    plt.legend()
    plt.savefig('Forecasting_Graphs/05_ARMA_Optimized.png', dpi=300)
    
    # Update leaderboard
    metrics_df = pd.DataFrame({
        'Model': ['ARMA(2,3)'], 
        'MAE': [mae], 
        'RMSE': [rmse], 
        'MAPE': [mape], 
        'R2': [r2]
    })
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, metrics_df], ignore_index=True).to_csv('leaderboard.csv', index=False)
    print(f"ARMA(2,3) Complete. MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    run_arma_optimized()

