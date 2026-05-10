import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set professional theme
sns.set_theme(style="whitegrid", context="talk")

def fix_stress_analysis():
    # 1. Load data
    data_path = 'data/wesad_forecasting_data.csv'
    df = pd.read_csv(data_path)
    
    # 2. Focus on Subject S2
    s2 = df[df['subject'] == 'S2'].copy()
    s2 = s2.sort_values('second')
    
    output_dir = 'EDA_Graphs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Create a Continuous Proxy for Stress (Rolling Mean)
    # The raw target_stress is binary (0/1), making ACF/PACF look like a block.
    # We apply a rolling window to see the "Stress Intensity" or "Density" over time.
    window_size = 60 # 1 minute window
    s2['stress_intensity'] = s2['target_stress'].rolling(window=window_size, center=True).mean().fillna(0)

    # 4. Graph: Professional Stress Timeline
    plt.figure(figsize=(16, 7))
    plt.plot(s2['second'], s2['target_stress'], color='gray', alpha=0.3, label='Raw Stress Binary')
    plt.plot(s2['second'], s2['stress_intensity'], color='#d62728', linewidth=2, label='Stress Intensity (Rolling Mean)')
    
    plt.fill_between(s2['second'], 0, s2['stress_intensity'], color='#d62728', alpha=0.1)
    
    plt.title('Subject S2: Temporal Stress Profile', fontweight='bold', fontsize=20)
    plt.xlabel('Time (Seconds)', fontweight='bold')
    plt.ylabel('Stress Score [0-1]', fontweight='bold')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.savefig(f'{output_dir}/01_S2_Stress_Timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 01_S2_Stress_Timeline.png")

    # 5. Graph: High-Resolution ACF & PACF (Fixed)
    # Using the continuous intensity makes the ACF/PACF meaningful for forecasting.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    plot_acf(s2['stress_intensity'], ax=axes[0], lags=100, color='#d62728', alpha=0.05)
    axes[0].set_title('ACF: Stress Temporal Dependency', fontweight='bold')
    axes[0].set_xlabel('Lag (Seconds)')
    
    plot_pacf(s2['stress_intensity'], ax=axes[1], lags=100, color='#d62728', alpha=0.05)
    axes[1].set_title('PACF: Partial Temporal Dependency', fontweight='bold')
    axes[1].set_xlabel('Lag (Seconds)')
    
    plt.suptitle('S2 Stress Diagnostic: Lag Correlation Analysis', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}/05_S2_Lag_Diagnostics.png', dpi=300)
    plt.close()
    print("Saved: 05_S2_Lag_Diagnostics.png")

    # 6. Graph: Differenced Stress (Stationary check)
    s2_diff = s2['stress_intensity'].diff().dropna()
    plt.figure(figsize=(16, 6))
    plt.plot(s2_diff, color='#ff7f0e', linewidth=1, label='Stationary Stress Delta')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('S2 Stress Intensity: Stationary Transformation (d=1)', fontweight='bold')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Change in Stress')
    plt.legend()
    plt.savefig(f'{output_dir}/06_S2_Stationary_Stress.png', dpi=300)
    plt.close()
    print("Saved: 06_S2_Stationary_Stress.png")

    # Save the processed S2 data for your verification
    s2[['second', 'target_stress', 'stress_intensity']].to_csv('S2_Diagnostic_Data.csv', index=False)
    print("Saved S2_Diagnostic_Data.csv for your verification.")

if __name__ == "__main__":
    fix_stress_analysis()
