import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import os

def run_generalization_test():
    # 1. Load data
    data_path = 'data/wesad_forecasting_data.csv'
    df_all = pd.read_csv(data_path)
    
    # 2. Prepare Subjects
    def prep_subj(subj_id):
        df = df_all[df_all['subject'] == subj_id].copy()
        # Use target_stress as score
        df['stress_score'] = df['target_stress'] 
        df['stress_lag1'] = df['stress_score'].shift(1)
        df['hr_lag1'] = df['eda_mean'] # Using eda_mean as proxy since HR isn't in this CSV
        df['eda_lag1'] = df['eda_mean'].shift(1)
        df['temp_lag1'] = df['temp_mean'].shift(1)
        return df.dropna().reset_index(drop=True)

    s2 = prep_subj('S2')
    s3 = prep_subj('S3')
    
    features = ['stress_lag1', 'eda_lag1', 'temp_lag1']
    
    # Train on S2
    X_train = s2[features].values
    y_train = s2['stress_score'].values
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    # Test on S3
    X_test_s3 = s3[features].values
    y_test_s3 = s3['stress_score'].values
    preds_s3 = model.predict(X_test_s3)
    
    # Test on S2 (held out portion)
    train_split = int(len(s2) * 0.8)
    preds_s2 = model.predict(s2[features].values[train_split:])
    mae_s2 = mean_absolute_error(s2['stress_score'].values[train_split:], preds_s2)
    mae_s3 = mean_absolute_error(y_test_s3, preds_s3)
    
    # 3. Visualization: Generalization Gap
    results = [
        {'Target': 'Internal (S2)', 'MAE': mae_s2},
        {'Target': 'Cross-Subject (S3)', 'MAE': mae_s3}
    ]
    df_res = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MAE', y='Target', data=df_res, palette='coolwarm', hue='Target', legend=False)
    plt.title('Applied Project: Cross-Subject Generalization (S2 vs S3)', fontweight='bold', fontsize=18)
    plt.xlabel('Mean Absolute Error (Lower is Better)')
    plt.tight_layout()
    plt.savefig('EDA_Graphs/17_Generalization_Test.png', dpi=300)
    plt.close()
    
    print(f"Generalization Test Complete. Graph saved: 17_Generalization_Test.png")
    print(f"  Internal MAE: {mae_s2:.4f}")
    print(f"  Cross-Subject MAE: {mae_s3:.4f}")

if __name__ == "__main__":
    run_generalization_test()
