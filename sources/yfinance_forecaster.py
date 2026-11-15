#--------------------------- импорт библиотек
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#--------------------------- параметры, с которыми можно поиграться
TICKER = "AAPL"             # можно заменить на "MSFT", "NVDA", "GOOGL" и т.д.
TRAIN_START = "2022-01-01" # начало периода, который видит модель
TRAIN_END = "2026-09-30"   # последняя дата обучения
FORECAST_END = "2026-10-11"
YFINANCE_END = "2026-10-12" # yfinance не включает end, поэтому берем следующий день

WINDOW = 100        # сколько прошлых дней берем для прогноза следующего дня
EPOCHS = 100       # сколько эпох обучать модели
BATCH_SIZE = 64
LR = 0.001         # learning rate

#--------------------------- загрузка котировок AAPL
cache_dir = Path(__file__).resolve().parents[2] / ".yfinance_cache"
cache_dir.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(cache_dir))

df = yf.download(
    TICKER,
    start=TRAIN_START,
    end=YFINANCE_END,
    interval="1d",
    auto_adjust=True,
    progress=False
)

prices = df["Close"].dropna()
if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]
print(prices.head())

#--------------------------- sliding window, создание выборки
def make_windows(series, window):
    x, y = [], []
    for i in range(len(series) - window):
        x.append(series[i:i+window])
        y.append(series[i+window])
    return torch.stack(x), torch.stack(y)

#--------------------------- train до 30 сентября, реальные цены после 30 сентября
train_prices = prices.loc[TRAIN_START:TRAIN_END]
real_future = prices.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):FORECAST_END]

if len(train_prices) <= WINDOW:
    raise RuntimeError("Для обучения слишком мало данных")

if real_future.empty:
    raise RuntimeError("Нет фактических цен для сравнения с прогнозом")

#--------------------------- масштабирование цен
prices_np = train_prices.to_numpy().reshape(-1)
future_np = real_future.to_numpy().reshape(-1)

split_index = int(len(prices_np)*0.8)
train_raw = prices_np[:split_index]

mean = train_raw.mean()
std = train_raw.std()
scaled = (prices_np-mean)/std

series_stock = torch.tensor(scaled,dtype=torch.float32)
X_stock,y_stock = make_windows(series_stock,WINDOW)

#--------------------------- делим обучающий период на train/test для проверки моделей
split_stock = int(len(X_stock)*0.8)
Xs_train,ys_train = X_stock[:split_stock],y_stock[:split_stock]
Xs_test,ys_test = X_stock[split_stock:],y_stock[split_stock:]

# Дальше тот же pipeline:
# baseline -> MLP -> LSTM -> MSE / MAE
#--------------------------- baseline
criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()

baseline = Xs_test[:,-1]
baseline_mse = criterion(baseline,ys_test).item()
baseline_mae = mae_criterion(baseline,ys_test).item()

#--------------------------- MLP
mlp_loader = DataLoader(
    TensorDataset(Xs_train,ys_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class PriceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WINDOW,1024),nn.ReLU(),nn.Linear(1024,1))
    def forward(self,x):
        return self.net(x).squeeze(1)

mlp_model = PriceMLP()
optimizer = torch.optim.Adam(mlp_model.parameters(),lr=LR)

for epoch in range(EPOCHS):
    mlp_model.train()
    for xb,yb in mlp_loader:
        optimizer.zero_grad()
        pred = mlp_model(xb)
        loss = criterion(pred,yb)
        loss.backward()
        optimizer.step()

mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(Xs_test)

mlp_mse = criterion(mlp_pred,ys_test).item()
mlp_mae = mae_criterion(mlp_pred,ys_test).item()

#--------------------------- LSTM
Xs_train_lstm = Xs_train.unsqueeze(-1)
Xs_test_lstm = Xs_test.unsqueeze(-1)

lstm_loader = DataLoader(
    TensorDataset(Xs_train_lstm,ys_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class PriceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1,hidden_size=216,batch_first=True)
        self.fc = nn.Linear(216,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(1)

lstm_model = PriceLSTM()
optimizer = torch.optim.Adam(lstm_model.parameters(),lr=LR)

for epoch in range(EPOCHS):
    lstm_model.train()
    for xb,yb in lstm_loader:
        optimizer.zero_grad()
        pred = lstm_model(xb)
        loss = criterion(pred,yb)
        loss.backward()
        optimizer.step()

lstm_model.eval()
with torch.no_grad():
    lstm_pred = lstm_model(Xs_test_lstm)

lstm_mse = criterion(lstm_pred,ys_test).item()
lstm_mae = mae_criterion(lstm_pred,ys_test).item()

print("Test on train period")
print("Baseline MSE:",baseline_mse,"MAE:",baseline_mae)
print("MLP MSE:",mlp_mse,"MAE:",mlp_mae)
print("LSTM MSE:",lstm_mse,"MAE:",lstm_mae)

#--------------------------- рекурсивный прогноз до 11 октября
steps = len(real_future)

window = series_stock[-WINDOW:].clone()
future_mlp = []
mlp_model.eval()
with torch.no_grad():
    for _ in range(steps):
        nv = mlp_model(window.unsqueeze(0))[0]
        future_mlp.append(nv.item())
        window = torch.cat([window[1:],nv.unsqueeze(0)])

window = series_stock[-WINDOW:].clone()
future_lstm = []
lstm_model.eval()
with torch.no_grad():
    for _ in range(steps):
        x = window.reshape(1,WINDOW,1)
        nv = lstm_model(x)[0]
        future_lstm.append(nv.item())
        window = torch.cat([window[1:],nv.unsqueeze(0)])

#--------------------------- перевод прогноза обратно в цены
future_mlp = np.array(future_mlp)*std + mean
future_lstm = np.array(future_lstm)*std + mean
future_baseline = np.repeat(prices_np[-1],steps)

baseline_real_mse = np.mean((future_np-future_baseline)**2)
baseline_real_mae = np.mean(np.abs(future_np-future_baseline))
mlp_real_mse = np.mean((future_np-future_mlp)**2)
mlp_real_mae = np.mean(np.abs(future_np-future_mlp))
lstm_real_mse = np.mean((future_np-future_lstm)**2)
lstm_real_mae = np.mean(np.abs(future_np-future_lstm))

#--------------------------- сравнение прогноза и реальности
comparison = pd.DataFrame(
    {
        "Real": future_np,
        "Baseline": future_baseline,
        "MLP": future_mlp,
        "LSTM": future_lstm,
        "MLP error": future_np-future_mlp,
        "LSTM error": future_np-future_lstm,
    },
    index=real_future.index
)

print()
print("Forecast from",TRAIN_END,"to",FORECAST_END)
print(comparison.round(2))
print()
print("Real Baseline MSE:",baseline_real_mse,"MAE:",baseline_real_mae)
print("Real MLP MSE:",mlp_real_mse,"MAE:",mlp_real_mae)
print("Real LSTM MSE:",lstm_real_mse,"MAE:",lstm_real_mae)

#--------------------------- график прогноза и реальных цен
history = 40
plot_dates = train_prices.index[-1:].append(real_future.index)
last_known_price = prices_np[-1]
real_plot = np.concatenate([[last_known_price],future_np])
baseline_plot = np.concatenate([[last_known_price],future_baseline])
mlp_plot = np.concatenate([[last_known_price],future_mlp])
lstm_plot = np.concatenate([[last_known_price],future_lstm])

plt.figure(figsize=(11,6))
plt.plot(
    train_prices.index[-history:],
    train_prices[-history:].to_numpy(),
    label="Known"
)
plt.plot(
    plot_dates,
    real_plot,
    marker="o",
    label="Real"
)
plt.plot(
    plot_dates,
    baseline_plot,
    marker="o",
    label="Baseline"
)
plt.plot(
    plot_dates,
    mlp_plot,
    marker="o",
    label="MLP Forecast"
)
plt.plot(
    plot_dates,
    lstm_plot,
    marker="o",
    label="LSTM Forecast"
)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
