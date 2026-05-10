import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Handle zero stress scores by adding small epsilon
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_sarima_optimized():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    series = df['stress_score'].values
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # 2. SARIMA Parameters
    # (p,d,q) = (2,1,3) from AIC
    # (P,D,Q,s) = (1,1,1,7) - common seasonal baseline for daily/weekly cycles
    history = list(train)
    predictions = []
    
    print("Running SARIMA(2,1,3)(1,1,1,7) Rolling Forecast...")
    # Process in batches or limited window for speed as SARIMA is heavy
    for i in range(len(test)):
        # Using a 100-point window for seasonal context
        model = SARIMAX(history[-100:], 
                        order=(2, 1, 3), 
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(test[i])
        
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(test)}")
    
    # 3. Comprehensive Metrics
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    r2 = r2_score(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)

    # 4. Professional Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(test, color='black', label='Observed Stress', alpha=0.7)
    plt.plot(predictions, color='#8c564b', linestyle='--', label='SARIMA(2,1,3)(1,1,1,7) Forecast')
    plt.title('Seasonal AutoRegressive Integrated Moving Average (SARIMA)', fontweight='bold')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Stress Score')
    
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nR2: {r2:.4f}'
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('Forecasting_Graphs/06_SARIMA_Optimized.png', dpi=300)
    
    # 5. Export Metrics CSV
    model_name = "SARIMA"
    metrics_data = {
        'Model': [model_name],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'R2': [r2]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f'metrics_{model_name.lower()}.csv', index=False)
    
    # Update master leaderboard
    try:
        leaderboard = pd.read_csv('leaderboard.csv')
        # Add MAPE and R2 if missing from old entries
        if 'MAPE' not in leaderboard.columns: leaderboard['MAPE'] = np.nan
        if 'R2' not in leaderboard.columns: leaderboard['R2'] = np.nan
        leaderboard = pd.concat([leaderboard, metrics_df], ignore_index=True)
        leaderboard.to_csv('leaderboard.csv', index=False)
    except:
        metrics_df.to_csv('leaderboard.csv', index=False)

    print(f"SARIMA Complete. MAE: {mae:.4f}")

if __name__ == "__main__":
    run_sarima_optimized()
