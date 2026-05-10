import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import os

def run_horizon_decay():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Features
    df['stress_lag1'] = df['stress_score'].shift(1)
    df['hr_lag1'] = df['heart_rate'].shift(1)
    df['eda_lag1'] = df['eda'].shift(1)
    df['temp_lag1'] = df['temp'].shift(1)
    
    horizons = [1, 5, 10, 30]
    results = []
    output_dir = 'EDA_Graphs'

    print("Running Forecast Horizon Decay Analysis...")
    for h in horizons:
        # Create target shifted by h steps
        df[f'target_h{h}'] = df['stress_score'].shift(-h)
        temp_df = df.dropna().reset_index(drop=True)
        
        features = ['stress_lag1', 'hr_lag1', 'eda_lag1', 'temp_lag1']
        X = temp_df[features].values
        y = temp_df[f'target_h{h}'].values
        
        train_size = int(len(y) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        results.append({'Horizon': f'{h}s', 'MAE': mae})
        print(f"  Horizon {h}s: MAE = {mae:.4f}")

    # 3. Visualization: Decay Curve
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Horizon', y='MAE', data=df_res, marker='o', color='red', linewidth=2.5)
    plt.title('Applied Project: Forecast Horizon Decay (Accuracy vs. Time)', fontweight='bold', fontsize=18)
    plt.xlabel('Forecasting Horizon (Seconds into Future)')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/15_Horizon_Decay_Analysis.png', dpi=300)
    plt.close()
    
    print(f"Horizon Analysis Complete. Graph saved: 15_Horizon_Decay_Analysis.png")

if __name__ == "__main__":
    run_horizon_decay()
