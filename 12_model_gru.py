import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

# Set professional theme
plt.style.use('seaborn-v0_8-whitegrid')

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

class StressGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StressGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def run_gru_forecast():
    # 1. Load Data
    df = pd.read_csv('s2_proper_continuous.csv')
    
    # 2. Data Preparation
    features = ['stress_score', 'heart_rate', 'eda', 'temp']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)

    SEQ_LENGTH = 10
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    train_size = int(len(y) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 3. Model Setup
    model = StressGRU(input_size=len(features), hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    print("Training GRU Neural Network...")
    start_time = time.time()
    epochs = 50
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            
    # 5. Forecasting
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test_t).numpy()
    
    dummy = np.zeros((len(predictions_scaled), len(features)))
    dummy[:, 0] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(features)))
    dummy_actual[:, 0] = y_test.flatten()
    actual = scaler.inverse_transform(dummy_actual)[:, 0]
    
    end_time = time.time()
    
    # 6. Metrics
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions)

    # 7. Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(actual, color='black', label='Actual Stress', alpha=0.7)
    plt.plot(predictions, color='#d62728', linestyle='--', label='GRU Forecast')
    plt.title('Deep Learning: GRU Neural Network Stress Forecasting', fontweight='bold')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Stress Score')
    
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nR2: {r2:.4f}'
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('Forecasting_Graphs/12_GRU_Forecast.png', dpi=300)
    plt.close()
    
    # 8. Export Metrics
    model_name = "GRU"
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'R2': [r2]
    })
    metrics_df.to_csv(f'metrics_gru.csv', index=False)
    
    # Update master leaderboard
    leaderboard = pd.read_csv('leaderboard.csv')
    pd.concat([leaderboard, metrics_df], ignore_index=True).to_csv('leaderboard.csv', index=False)

    print(f"GRU Complete. Time: {end_time - start_time:.2f}s, MAE: {mae:.4f}")

if __name__ == "__main__":
    run_gru_forecast()
