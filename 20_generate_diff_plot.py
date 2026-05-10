import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set professional theme
sns.set_theme(style="whitegrid", context="talk")

def generate_differenced_plot():
    # 1. Load data
    data_path = 's2_proper_continuous.csv'
    if not os.path.exists(data_path):
        print("Data not found.")
        return
    
    df = pd.read_csv(data_path)
    series = df['stress_score']
    
    # 2. Apply First-Order Differencing
    diff_series = series.diff().dropna()
    
    # 3. Visualization
    plt.figure(figsize=(16, 6))
    plt.plot(df['second'][1:], diff_series, color='#ff7f0e', linewidth=1, label='Stationary Delta (d=1)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.title('Applied Project: First-Order Differenced Stress Signal (Stationarity Proof)', fontweight='bold', fontsize=20)
    plt.xlabel('Time (Seconds)', fontweight='bold')
    plt.ylabel('Change in Stress Intensity', fontweight='bold')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('EDA_Graphs/05_Differenced_Signal_Timeline.png', dpi=300)
    plt.close()
    print("Saved: 05_Differenced_Signal_Timeline.png")

if __name__ == "__main__":
    generate_differenced_plot()
