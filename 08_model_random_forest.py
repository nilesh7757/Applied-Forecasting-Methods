import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def run_random_forest_forecast():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Feature Engineering: Lagged Features (Avoid Leakage)
    # Using t-1 and t-2 to predict t
    df['stress_lag1'] = df['stress_score'].shift(1)
    df['stress_lag2'] = df['stress_score'].shift(2)
    df['hr_lag1'] = df['heart_rate'].shift(1)
    df['eda_lag1'] = df['eda'].shift(1)
    df['temp_lag1'] = df['temp'].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    features = ['stress_lag1', 'stress_lag2', 'hr_lag1', 'eda_lag1', 'temp_lag1']
    X = df[features].values
    y = df['stress_score'].values
    
    # 3. Train/Test Split
    train_size = int(len(y) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Training Random Forest Regressor...")
    start_time = time.time()
    
    # Fit the model once for this phase
    # (In high-frequency data, RF is fast enough to retrain, 
    # but a single fit is standard for ML baseline)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 4. Rolling Forecast Simulation
    # Even though it's ML, we simulate rolling to ensure no leakage
    predictions = rf.predict(X_test)
    
    end_time = time.time()
    
    # 5. Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    # 6. Professional Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='black', label='Observed Stress', alpha=0.7)
    plt.plot(predictions, color='#17becf', linestyle='--', label='Random Forest Forecast')
    plt.title('Machine Learning: Random Forest Stress Forecasting', fontweight='bold')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Stress Score')
    
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nR2: {r2:.4f}'
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('Forecasting_Graphs/08_Random_Forest_Forecast.png', dpi=300)
    
    # 7. Export Metrics
    model_name = "Random Forest"
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'R2': [r2]
    })
    metrics_df.to_csv(f'metrics_random_forest.csv', index=False)
    
    # Update master leaderboard
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, metrics_df], ignore_index=True).to_csv('leaderboard.csv', index=False)

    print(f"Random Forest Complete. Time: {end_time - start_time:.2f}s, MAE: {mae:.4f}")

if __name__ == "__main__":
    run_random_forest_forecast()
