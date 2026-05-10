import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

def run_quantile_regression():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Feature Engineering: Lagged Features
    df['stress_lag1'] = df['stress_score'].shift(1)
    df['hr_lag1'] = df['heart_rate'].shift(1)
    df['eda_lag1'] = df['eda'].shift(1)
    df['temp_lag1'] = df['temp'].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    features = ['stress_lag1', 'hr_lag1', 'eda_lag1', 'temp_lag1']
    X = df[features].values
    y = df['stress_score'].values
    
    # Train/Test Split
    train_size = int(len(y) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_test = y[train_size:]
    
    output_dir = 'Forecasting_Graphs'
    
    # 3. Fit Quantile Regressors (5th, 50th, 95th percentiles)
    quantiles = [0.05, 0.50, 0.95]
    preds = {}
    
    print("Training Quantile Regressors (Risk Boundaries)...")
    for q in quantiles:
        model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, random_state=42)
        model.fit(X[:train_size], y[:train_size])
        preds[q] = model.predict(X_test)
        print(f"  Completed quantile: {q}")

    # 4. Professional Visualization: Shaded Risk Boundary
    plt.figure(figsize=(16, 8))
    
    # Plot Actual
    plt.plot(y_test, color='black', label='Actual Stress', alpha=0.6, linewidth=1.5)
    
    # Plot Median (50th)
    plt.plot(preds[0.5], color='blue', label='Median Forecast (50th)', linestyle='--', linewidth=2)
    
    # Shade the 5th-95th Risk Interval
    plt.fill_between(range(len(y_test)), preds[0.05], preds[0.95], 
                     color='orange', alpha=0.3, label='90% Risk Confidence Interval')
    
    # Highlight the 95th Percentile as the "Upper Risk Boundary"
    plt.plot(preds[0.95], color='red', label='Upper Risk Boundary (95th)', alpha=0.8, linewidth=1)

    plt.title('Applied Project: Stress Risk Boundary Forecasting (Quantile Regression)', fontweight='bold', fontsize=20)
    plt.xlabel('Time (Seconds)', fontweight='bold')
    plt.ylabel('Stress Score', fontweight='bold')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/13_Quantile_Risk_Boundaries.png', dpi=300)
    plt.close()
    
    # 5. Export Metrics (Using Median for comparison)
    mae = mean_absolute_error(y_test, preds[0.5])
    # For Quantiles, we look at "Pinball Loss" but for leaderboard we use MAE of median
    metrics_df = pd.DataFrame({
        'Model': ['Quantile Regression (Median)'],
        'MAE': [mae],
        'RMSE': [np.nan], # Not standard for quantile
        'MAPE': [np.nan],
        'R2': [np.nan]
    })
    
    # Append to leaderboard
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, metrics_df], ignore_index=True).to_csv('leaderboard.csv', index=False)
    
    print(f"Quantile Model Complete. Graph saved: 13_Quantile_Risk_Boundaries.png")

if __name__ == "__main__":
    run_quantile_regression()
