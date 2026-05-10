import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set professional theme
sns.set_theme(style="whitegrid", context="talk")

def generate_multi_subject_eda():
    # 1. Load data
    data_path = 'data/wesad_forecasting_data.csv'
    if not os.path.exists(data_path):
        print("Main dataset not found.")
        return
    
    df = pd.read_csv(data_path)
    output_dir = 'EDA_Graphs'
    
    # We will plot S3 and S4 to show dataset diversity
    for subj in ['S3', 'S4']:
        subj_df = df[df['subject'] == subj].reset_index()
        
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # Plotting Physiological Signals
        ax1.plot(subj_df['second'], subj_df['eda_mean'], label='EDA (Electrodermal Activity)', color='#1f77b4', linewidth=1.5, alpha=0.8)
        ax1.plot(subj_df['second'], subj_df['temp_mean'], label='Skin Temperature', color='#2ca02c', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Time (Seconds)', fontweight='bold')
        ax1.set_ylabel('Physiological Signals', color='black', fontweight='bold')
        
        # Secondary axis for Stress Label
        ax2 = ax1.twinx()
        ax2.fill_between(subj_df['second'], 0, subj_df['target_stress'], color='#d62728', alpha=0.2, label='Stress Area')
        ax2.plot(subj_df['second'], subj_df['target_stress'], color='#d62728', linewidth=1, linestyle='--')
        ax2.set_ylabel('Binary Stress Label', color='#d62728', fontweight='bold')
        
        plt.title(f'Dataset Audit: Subject {subj} Timeline', pad=20)
        ax1.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_Subject_{subj}_Timeline.png', dpi=300)
        plt.close()
        print(f"Saved: 01_Subject_{subj}_Timeline.png")

if __name__ == "__main__":
    generate_multi_subject_eda()
