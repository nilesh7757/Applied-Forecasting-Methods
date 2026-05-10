import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set professional theme
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def run_proper_eda():
    # 1. Load the proper continuous data
    data_path = 's2_proper_continuous.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows")
    
    output_dir = 'EDA_Graphs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Graph 1: Subject S2 Proper Stress Timeline
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plotting Physiological Signals
    ax1.plot(df['second'], df['eda'], label='EDA (Electrodermal Activity)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    ax1.plot(df['second'], df['temp'], label='Skin Temperature', color='#2ca02c', linewidth=1.5, alpha=0.7)
    ax1.plot(df['second'], df['heart_rate'], label='Heart Rate', color='#9467bd', linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('Time (Seconds)', fontweight='bold')
    ax1.set_ylabel('Physiological Signals', fontweight='bold')
    
    # Secondary axis for Continuous Stress Score
    ax2 = ax1.twinx()
    fill = ax2.fill_between(df['second'], 0, df['stress_score'], color='#d62728', alpha=0.2, label='Continuous Stress Score')
    line_stress, = ax2.plot(df['second'], df['stress_score'], color='#d62728', linewidth=2, label='Stress Score (Target)')
    ax2.set_ylabel('Stress Score', color='#d62728', fontweight='bold')
    
    # Custom Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + [line_stress], labels_1 + ['Stress Score'], loc='upper right', frameon=True, shadow=True)
    
    plt.title('Subject S2: High-Resolution Continuous Stress Profile', pad=20)
    plt.savefig(f'{output_dir}/01_S2_Proper_Timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 01_S2_Proper_Timeline.png")

    # 3. Graph 2: Stationarity & Diagnostics
    # ADF Test
    result = adfuller(df['stress_score'].dropna())
    p_val = result[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_acf(df['stress_score'], ax=axes[0], lags=50, color='#d62728', alpha=0.05)
    axes[0].set_title(f'ACF: Continuous Stress (ADF p={p_val:.4f})', fontweight='bold')
    
    plot_pacf(df['stress_score'], ax=axes[1], lags=50, color='#d62728', alpha=0.05)
    axes[1].set_title('PACF: Continuous Stress', fontweight='bold')
    
    plt.suptitle('Subject S2: Statistical Memory & Stationarity Diagnostics', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}/02_S2_Proper_Diagnostics.png', dpi=300)
    plt.close()
    print("Saved: 02_S2_Proper_Diagnostics.png")

    # 4. Graph 3: Feature Correlations (Full Matrix)
    plt.figure(figsize=(12, 10))
    corr = df[['heart_rate', 'eda', 'temp', 'stress_score']].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f", center=0, square=True, linewidths=0.5)
    plt.title('Full Inter-Signal Correlation Matrix (Physiology vs Stress)', fontweight='bold', fontsize=16)
    plt.savefig(f'{output_dir}/03_S2_Proper_Correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 03_S2_Proper_Correlation.png")

    # 4b. New Graph: Stress vs EDA Joint Analysis (Pairplot)
    plt.figure(figsize=(15, 15))
    pair_plot = sns.pairplot(df[['heart_rate', 'eda', 'temp', 'stress_score']], kind='reg', diag_kind='kde',
                             plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    pair_plot.fig.suptitle('Multimodal Signal Distributions & Bivariate Stress Relationships', y=1.02, fontsize=20, fontweight='bold')
    pair_plot.savefig(f'{output_dir}/03b_Bivariate_Relationships.png', dpi=300)
    plt.close()
    print("Saved: 03b_Bivariate_Relationships.png")

    # 5. Graph 4: Feature Distributions with KDE
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.histplot(df['eda'], kde=True, color='#1f77b4', ax=axes[0])
    axes[0].set_title('EDA Distribution', fontweight='bold')
    
    sns.histplot(df['temp'], kde=True, color='#2ca02c', ax=axes[1])
    axes[1].set_title('Temperature Distribution', fontweight='bold')

    sns.histplot(df['heart_rate'], kde=True, color='#9467bd', ax=axes[2])
    axes[2].set_title('Heart Rate Distribution', fontweight='bold')
    
    plt.suptitle('Physiological Feature Variance & Density Audit', fontsize=22, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_Feature_Density_Analysis.png', dpi=300)
    plt.close()
    print("Saved: 02_Feature_Density_Analysis.png")


if __name__ == "__main__":
    run_proper_eda()
