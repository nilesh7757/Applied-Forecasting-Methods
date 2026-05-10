import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set professional style
sns.set_theme(style="whitegrid", context="talk")

def run_advanced_diagnostics():
    # 1. Load data
    data_path = 's2_proper_continuous.csv'
    if not os.path.exists(data_path): return
    df = pd.read_csv(data_path)
    
    series = df['stress_score'].dropna()
    output_dir = 'EDA_Graphs'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 2. ADF Test on Raw Data
    res_orig = adfuller(series)
    p_orig = res_orig[1]
    
    # 3. ADF Test on Differenced Data (d=1)
    diff_series = series.diff().dropna()
    res_diff = adfuller(diff_series)
    p_diff = res_diff[1]

    # 4. Plot 1: Raw Signal Diagnostics (Why it looks "wrong")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_acf(series, ax=axes[0], lags=50, color='#1f77b4')
    axes[0].set_title(f'Raw ACF (Non-Stationary, p={p_orig:.3f})', fontweight='bold')
    axes[0].set_ylabel('Correlation')
    
    plot_pacf(series, ax=axes[1], lags=50, color='#1f77b4', method='ywm')
    axes[1].set_title('Raw PACF', fontweight='bold')
    
    plt.suptitle('Diagnostic A: Raw Stress Signal (Slow Decay = Non-Stationary)', fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_Raw_Lag_Diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Plot 2: Differenced Signal Diagnostics (The "Right" way)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_acf(diff_series, ax=axes[0], lags=50, color='#ff7f0e')
    axes[0].set_title(f'Stationary ACF (d=1, p={p_diff:.2e})', fontweight='bold')
    axes[0].set_ylabel('Correlation')
    
    plot_pacf(diff_series, ax=axes[1], lags=50, color='#ff7f0e', method='ywm')
    axes[1].set_title('Stationary PACF (d=1)', fontweight='bold')
    
    plt.suptitle('Diagnostic B: Differenced Stress Signal (Proper for Model Selection)', fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_Stationary_Lag_Diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Diagnostics complete.")
    print(f"Raw p-value: {p_orig:.4f} (Non-Stationary)")
    print(f"Differenced p-value: {p_diff:.4e} (Stationary)")
    print("Check 05_Raw_Lag_Diagnostics.png and 06_Stationary_Lag_Diagnostics.png")

if __name__ == "__main__":
    run_advanced_diagnostics()


