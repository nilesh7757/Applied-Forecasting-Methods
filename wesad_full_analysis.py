import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("WESAD STRESS PREDICTION ANALYSIS")
print("="*70)

# Load WESAD data
def load_wesad_stress_series():
    wesad_dir = 'D:/Aplied/WESAD/WESAD'
    subjects = sorted([d for d in os.listdir(wesad_dir) if d.startswith('S')])
    
    # Load first subject for time series analysis
    subj = subjects[0]
    pkl_path = os.path.join(wesad_dir, subj, f'{subj}.pkl')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    labels = data['label']
    chest = data['signal']['chest']
    
    # Respiration signal
    resp = chest['Resp']
    
    # Get continuous stress level (interpolate labels to match signal length)
    # Labels: 0=undefined, 1=stress, 2=amusement, 3=meditation
    # Create binary stress: 1 if stress, 0 otherwise
    label_len = len(labels)
    signal_len = len(resp)
    
    # Downsample to match
    factor = signal_len // label_len
    stress_indicator = np.repeat(labels, factor)[:signal_len]
    
    # Create stress time series (1=stress, 0=no stress)
    stress_series = (stress_indicator == 1).astype(int)
    
    # For analysis, use respiration as the signal
    # Downsample for computational efficiency
    sample_step = 1000
    stress_ts = stress_series[::sample_step]
    resp_ts = resp[::sample_step]
    
    return stress_ts, resp_ts, subj

print("\nLoading WESAD subject data...")
stress_ts, resp_ts, subj = load_wesad_stress_series()
print(f"Subject: {subj}, Length: {len(stress_ts)}")
print(f"Stress samples: {stress_ts.sum()}, No stress: {len(stress_ts) - stress_ts.sum()}")

# ============================================
# 1. Time Series Plot
# ============================================
print("\n" + "="*70)
print("1. TIME SERIES PLOT")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Stress level time series
axes[0].plot(stress_ts, 'b-', linewidth=0.8, alpha=0.7)
axes[0].fill_between(range(len(stress_ts)), stress_ts, alpha=0.3)
axes[0].set_title(f'WESAD Subject {subj} - Stress Level Time Series', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time Index')
axes[0].set_ylabel('Stress (0=No, 1=Yes)')
axes[0].set_ylim(-0.1, 1.1)
axes[0].grid(True, alpha=0.3)

# Respiration time series
axes[1].plot(resp_ts, 'g-', linewidth=0.5, alpha=0.7)
axes[1].set_title(f'WESAD Subject {subj} - Respiration Signal', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time Index')
axes[1].set_ylabel('Respiration')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/Aplied/wesad_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wesad_timeseries.png")

# ============================================
# 2. ADF Test
# ============================================
print("\n" + "="*70)
print("2. AUGMENTED DICKEY-FULLER (ADF) TEST")
print("="*70)

result = adfuller(stress_ts)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.6f}")
print(f"Critical Values:")
for key, val in result[4].items():
    print(f"  {key}: {val:.4f}")
print(f"\nConclusion: {'STATIONARY' if result[1] < 0.05 else 'NON-STATIONARY'}")

fig, ax = plt.subplots(figsize=(8, 6))
textstr = f'ADF Test Results\n{"="*40}\n\nADF Statistic: {result[0]:.4f}\np-value: {result[1]:.6f}\n\nCritical Values:\n  1%: {result[4]["1%"]:.4f}\n  5%: {result[4]["5%"]:.4f}\n  10%: {result[4]["10%"]:.4f}\n\n{"="*40}\nConclusion: {"STATIONARY" if result[1] < 0.05 else "NON-STATIONARY"}'
ax.text(0.5, 0.5, textstr, fontsize=12, family='monospace',
       ha='center', va='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightgreen' if result[1] < 0.05 else 'lightyellow', alpha=0.9))
ax.axis('off')
ax.set_title('WESAD - ADF Test', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/Aplied/wesad_adf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wesad_adf.png")

# ============================================
# 3. ACF and PACF Plots
# ============================================
print("\n" + "="*70)
print("3. ACF AND PACF PLOTS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(stress_ts, ax=axes[0], lags=20, alpha=0.05)
axes[0].set_title('WESAD - Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Autocorrelation')

plot_pacf(stress_ts, ax=axes[1], lags=20, alpha=0.05, method='ywm')
axes[1].set_title('WESAD - Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Partial Autocorrelation')

plt.tight_layout()
plt.savefig('D:/Aplied/wesad_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wesad_acf_pacf.png")

# Get ACF/PACF values
acf_vals = acf(stress_ts, nlags=5)
pacf_vals = pacf(stress_ts, nlags=5, method='ywm')
print(f"\nACF (first 5 lags): {acf_vals[:5].round(4)}")
print(f"PACF (first 5 lags): {pacf_vals[:5].round(4)}")

# ============================================
# Train/Test Split for Models
# ============================================
split_idx = int(len(stress_ts) * 0.8)
train = stress_ts[:split_idx]
test = stress_ts[split_idx:]

print(f"\nTrain: {len(train)}, Test: {len(test)}")

# ============================================
# 4. AR, MA, ARMA, ARIMA Models
# ============================================
print("\n" + "="*70)
print("4. AR, MA, ARMA, ARIMA MODELS")
print("="*70)

results = []
predictions = {}

# AR(1)
print("\nFitting AR(1)...")
try:
    ar_model = ARIMA(train, order=(1, 0, 0))
    ar_fit = ar_model.fit()
    ar_pred = ar_fit.forecast(steps=len(test))
    predictions['AR(1)'] = ar_pred
    mse = mean_squared_error(test, ar_pred)
    results.append(('AR(1)', mse, np.sqrt(mse)))
    print(f"  MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}")
except Exception as e:
    print(f"  Error: {e}")

# MA(1)
print("Fitting MA(1)...")
try:
    ma_model = ARIMA(train, order=(0, 0, 1))
    ma_fit = ma_model.fit()
    ma_pred = ma_fit.forecast(steps=len(test))
    predictions['MA(1)'] = ma_pred
    mse = mean_squared_error(test, ma_pred)
    results.append(('MA(1)', mse, np.sqrt(mse)))
    print(f"  MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ARMA(1,1)
print("Fitting ARMA(1,1)...")
try:
    arma_model = ARIMA(train, order=(1, 0, 1))
    arma_fit = arma_model.fit()
    arma_pred = arma_fit.forecast(steps=len(test))
    predictions['ARMA(1,1)'] = arma_pred
    mse = mean_squared_error(test, arma_pred)
    results.append(('ARMA(1,1)', mse, np.sqrt(mse)))
    print(f"  MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ARIMA(1,1,1)
print("Fitting ARIMA(1,1,1)...")
try:
    arima_model = ARIMA(train, order=(1, 1, 1))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=len(test))
    predictions['ARIMA(1,1,1)'] = arima_pred
    mse = mean_squared_error(test, arima_pred)
    results.append(('ARIMA(1,1,1)', mse, np.sqrt(mse)))
    print(f"  MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================
# 5. SARIMA Model
# ============================================
print("\n" + "="*70)
print("5. SARIMA MODEL")
print("="*70)

print("Fitting SARIMA(1,0,1)(1,0,1,7)...")
try:
    # SARIMA with weekly seasonality
    sarima_model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.forecast(steps=len(test))
    predictions['SARIMA'] = sarima_pred
    mse = mean_squared_error(test, sarima_pred)
    results.append(('SARIMA(1,0,1)(1,0,1,7)', mse, np.sqrt(mse)))
    print(f"  MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================
# Results Summary
# ============================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

df_results = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE'])
df_results = df_results.sort_values('RMSE')
print(df_results.to_string(index=False))

# Create comparison graph
test_idx = np.arange(len(train), len(stress_ts))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_idx, test, 'ko-', markersize=4, label='Actual', linewidth=1.5)

colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (name, pred) in enumerate(predictions.items()):
    ax.plot(test_idx, pred, '--', color=colors[i], linewidth=1.5, label=name, alpha=0.7)

ax.axhline(y=test.mean(), color='red', linestyle=':', alpha=0.5, label=f'Test Mean={test.mean():.2f}')
ax.axvline(x=split_idx, color='gray', linestyle=':', linewidth=2)
ax.set_title(f'WESAD - Model Predictions vs Actual', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Index')
ax.set_ylabel('Stress Level')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/Aplied/wesad_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: wesad_model_comparison.png")

# Metrics comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].barh(df_results['Model'], df_results['RMSE'], color='steelblue', alpha=0.8)
axes[0].set_xlabel('RMSE (Lower is Better)')
axes[0].set_title('WESAD - Model RMSE Comparison', fontweight='bold')
axes[0].invert_yaxis()

axes[1].barh(df_results['Model'], df_results['MSE'], color='coral', alpha=0.8)
axes[1].set_xlabel('MSE (Lower is Better)')
axes[1].set_title('WESAD - Model MSE Comparison', fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('D:/Aplied/wesad_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wesad_metrics.png")

best = df_results.iloc[0]
print(f"\n*** BEST MODEL: {best['Model']} with RMSE = {best['RMSE']:.4f} ***")

print("\n" + "="*70)
print("ALL WESAD ANALYSIS COMPLETE!")
print("="*70)
print("Generated files:")
print("  - wesad_timeseries.png")
print("  - wesad_adf.png")
print("  - wesad_acf_pacf.png")
print("  - wesad_model_comparison.png")
print("  - wesad_metrics.png")