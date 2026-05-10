import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def investigate_s2():
    wesad_dir = 'data/S2'
    pkl_path = os.path.join(wesad_dir, 'S2.pkl')
    
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    labels = data['label']
    chest = data['signal']['chest']
    eda = chest['EDA'].flatten()
    temp = chest['Temp'].flatten()
    resp = chest['Resp'].flatten()
    
    # Match lengths
    min_len = min(len(labels), len(eda))
    labels = labels[:min_len]
    eda = eda[:min_len]
    temp = temp[:min_len]
    resp = resp[:min_len]
    
    # Downsample for plotting (700Hz is too much)
    step = 700 
    time_sec = np.arange(len(labels))[::step] / 700
    labels_ds = labels[::step]
    eda_ds = eda[::step]
    temp_ds = temp[::step]
    resp_ds = resp[::step]

    # Plot everything
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # 1. Labels
    axes[0].plot(time_sec, labels_ds, color='black', label='Raw Labels (0-7)')
    axes[0].set_title('WESAD S2 - Raw Labels (0=None, 1=Baseline, 2=Stress, 3=Amusement, 4=Meditation)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. EDA
    axes[1].plot(time_sec, eda_ds, color='blue', label='EDA')
    axes[1].set_title('EDA (Electrodermal Activity)')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Temp
    axes[2].plot(time_sec, temp_ds, color='red', label='Skin Temp')
    axes[2].set_title('Skin Temperature')
    axes[2].legend()
    axes[2].grid(True)
    
    # 4. Resp
    axes[3].plot(time_sec, resp_ds, color='green', label='Respiration')
    axes[3].set_title('Respiration')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.xlabel('Time (Seconds)')
    plt.tight_layout()
    plt.savefig('S2_Raw_Investigation.png', dpi=300)
    plt.close()
    print("Saved: S2_Raw_Investigation.png")
    
    # Save a CSV of downsampled S2 data for inspection
    df_invest = pd.DataFrame({
        'second': time_sec,
        'label': labels_ds,
        'eda': eda_ds,
        'temp': temp_ds,
        'resp': resp_ds
    })
    df_invest.to_csv('S2_Raw_Investigation.csv', index=False)
    print("Saved: S2_Raw_Investigation.csv")

if __name__ == "__main__":
    investigate_s2()
