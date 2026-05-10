import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import os

def run_feature_ablation():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Feature Engineering: Lagged Features
    df['stress_lag1'] = df['stress_score'].shift(1)
    df['hr_lag1'] = df['heart_rate'].shift(1)
    df['eda_lag1'] = df['eda'].shift(1)
    df['temp_lag1'] = df['temp'].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    y = df['stress_score'].values
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Define Ablation Scenarios
    scenarios = {
        'Univariate (Stress Only)': ['stress_lag1'],
        'Physiological Only (HR+EDA+TEMP)': ['hr_lag1', 'eda_lag1', 'temp_lag1'],
        'Multivariate (Full Context)': ['stress_lag1', 'hr_lag1', 'eda_lag1', 'temp_lag1']
    }
    
    results = []
    output_dir = 'EDA_Graphs'
    
    print("Running Feature Ablation Study...")
    for name, features in scenarios.items():
        X = df[features].values
        X_train, X_test = X[:train_size], X[train_size:]
        
        # Using XGBoost for fast, reliable ablation
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        results.append({'Scenario': name, 'MAE': mae})
        print(f"  {name}: MAE = {mae:.4f}")

    # 3. Visualization: Ablation Comparison
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(12, 7))
    sns.barplot(x='MAE', y='Scenario', data=df_res, palette='magma', hue='Scenario', legend=False)
    plt.title('Applied Project: Feature Ablation Study (Importance of Context)', fontweight='bold', fontsize=18)
    plt.xlabel('Mean Absolute Error (Lower is Better)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/14_Feature_Ablation_Study.png', dpi=300)
    plt.close()
    
    print(f"Ablation Study Complete. Graph saved: 14_Feature_Ablation_Study.png")

if __name__ == "__main__":
    run_feature_ablation()
