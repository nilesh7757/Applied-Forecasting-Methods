import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_final_comparison():
    # 1. Load the accumulated leaderboard
    df = pd.read_csv('leaderboard.csv')
    
    # 2. Data Cleaning
    # Drop duplicates if any (e.g. from multiple runs)
    df = df.drop_duplicates(subset=['Model'], keep='last')
    
    # Clean up names for the report
    name_map = {
        'AR(2)': 'AR (p=2)',
        'MA(3)': 'MA (q=3)',
        'ARMA(2,3)': 'ARMA (2,3)',
        'SARIMA': 'SARIMA (2,1,3)(1,1,1,7)',
        'SARIMAX': 'SARIMAX (Exog Lags)',
        'Random Forest': 'Random Forest',
        'XGBoost': 'XGBoost',
        'LSTM': 'LSTM (DL)',
        'GRU': 'GRU (DL)',
        'SVR': 'SVR (ML)',
        'Naive (Baseline)': 'Naive (Baseline)',
        'SMA (Window=10)': 'SMA (Window=10)'
    }
    df['Model'] = df['Model'].map(lambda x: name_map.get(x, x))
    
    # Sort by MAE (Lower is better)
    df = df.sort_values('MAE', ascending=True).reset_index(drop=True)
    df.to_csv('final_model_leaderboard.csv', index=False)
    
    # 3. Final Comparison Visualization (Bar Charts)
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    titles = ['Mean Absolute Error (Lower is Better)', 
              'Root Mean Squared Error (Lower is Better)',
              'MAPE % (Lower is Better)',
              'R2 Score (Higher is Better)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        # For R2, higher is better, so we sort differently for plotting if needed
        # but keeping consistency with MAE sort is usually better for comparison
        sns.barplot(x=metric, y='Model', data=df, ax=ax, hue='Model', palette='viridis', legend=False)
        ax.set_title(titles[i], fontweight='bold', fontsize=16)
        ax.set_xlabel(metric)
        ax.set_ylabel('')

    plt.suptitle('Applied Project: Final Model Performance Comparison (Subject S2)', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Forecasting_Graphs/99_Final_Model_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Final Leaderboard and Comparison Chart Generated.")
    print("\n--- TOP 3 MODELS BY MAE ---")
    print(df[['Model', 'MAE', 'R2']].head(3))

if __name__ == "__main__":
    generate_final_comparison()
