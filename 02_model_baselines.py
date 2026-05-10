import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_baselines():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    series = df['stress_score'].values
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    output_dir = 'Forecasting_Graphs'
    
    # 2. Naive Model (Persistence)
    # Logic: Today's stress is tomorrow's forecast
    naive_preds = series[train_size-1 : -1]
    
    # 3. SMA Model (Simple Moving Average)
    # Logic: Mean of the last 10 seconds is the forecast
    sma_window = 10
    sma_preds = []
    history = list(train)
    for i in range(len(test)):
        yhat = np.mean(history[-sma_window:])
        sma_preds.append(yhat)
        history.append(test[i])
    sma_preds = np.array(sma_preds)

    # 4. Metrics & Export
    baselines = []
    for name, preds in [("Naive (Baseline)", naive_preds), ("SMA (Window=10)", sma_preds)]:
        mae = mean_absolute_error(test, preds)
        rmse = np.sqrt(mean_squared_error(test, preds))
        mape = mean_absolute_percentage_error(test, preds)
        r2 = r2_score(test, preds)
        
        baselines.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        })
        
        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(test, color='black', label='Actual', alpha=0.6)
        plt.plot(preds, label=f'{name} Forecast', linestyle='--')
        plt.title(f'Baseline Model: {name}', fontweight='bold')
        plt.legend()
        plt.savefig(f'{output_dir}/00_{name.replace(" ", "_")}.png', dpi=300)
        plt.close()

    # Save metrics
    df_baselines = pd.DataFrame(baselines)
    df_baselines.to_csv('metrics_baselines.csv', index=False)
    
    # Update Leaderboard
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, df_baselines], ignore_index=True).to_csv('leaderboard.csv', index=False)
    print("Naive and SMA Baselines complete.")

if __name__ == "__main__":
    run_baselines()
