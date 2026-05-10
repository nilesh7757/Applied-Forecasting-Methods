import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_wesad_data():
    wesad_dir = 'D:/Aplied/WESAD/WESAD'
    if not os.path.exists(wesad_dir):
        print("WESAD folder not found, trying to extract...")
        import zipfile
        with zipfile.ZipFile('D:/Aplied/WESAD.zip', 'r') as z:
            z.extractall('D:/Aplied/WESAD')
        wesad_dir = 'D:/Aplied/WESAD/WESAD'
    
    subjects = sorted([d for d in os.listdir(wesad_dir) if d.startswith('S')])
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    all_data = []
    for subj in subjects[:5]:  # Use first 5 subjects for faster processing
        pkl_path = os.path.join(wesad_dir, subj, f'{subj}.pkl')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                if 'label' in data and 'signal' in data:
                    labels = data['label']
                    if hasattr(labels, 'reshape'):
                        # Get stress label (typically label 1 = stress)
                        stress_mask = (labels == 1) | (labels == 2)  # stress or amusement
                        if 'chest' in data['signal'] and 'Resp' in data['signal']['chest']:
                            resp = data['signal']['chest']['Resp']
                            # Sample to match labels
                            min_len = min(len(resp), len(labels))
                            stress_levels = labels[:min_len]
                            resp_signal = resp[:min_len]
                            
                            for i in range(0, min_len-100, 100):  # Sample every 100 points
                                all_data.append({
                                    'subject': subj,
                                    'resp': resp_signal[i:i+100].mean(),
                                    'label': stress_levels[i]
                                })
            except Exception as e:
                print(f"Error loading {subj}: {e}")
    
    return pd.DataFrame(all_data)

print("Loading WESAD data...")
df = load_wesad_data()
print(f"Loaded {len(df)} samples")
print(df.head())
print(f"\nLabel distribution:\n{df['label'].value_counts()}")