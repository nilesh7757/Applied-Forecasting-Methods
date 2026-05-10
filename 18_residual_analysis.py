import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import os

def run_residual_analysis():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Features
    df['stress_lag1'] = df['stress_score'].shift(1)
    df['hr_lag1'] = df['heart_rate'].shift(1)
    df['eda_lag1'] = df['eda'].shift(1)
    df['temp_lag1'] = df['temp'].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    features = ['stress_lag1', 'hr_lag1', 'eda_lag1', 'temp_lag1']
    X = df[features].values
    y = df['stress_score'].values
    
    train_size = int(len(y) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train Best ML model for residuals
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # 3. Calculate Residuals
    residuals = y_test - preds
    output_dir = 'EDA_Graphs'

    # 4. Visualization: Residual Distribution
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Histogram
    sns.histplot(residuals, kde=True, ax=axes[0], color='purple')
    axes[0].set_title('Residuals Distribution (Error Histogram)', fontweight='bold')
    axes[0].set_xlabel('Prediction Error')
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot (Gaussianity Check)', fontweight='bold')
    
    plt.suptitle('Applied Project: Residual Diagnostics (Error Analysis)', fontsize=22, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/16_Residual_Analysis.png', dpi=300)
    plt.close()
    
    print(f"Residual Analysis Complete. Graph saved: 16_Residual_Analysis.png")

if __name__ == "__main__":
    run_residual_analysis()
