#--------------------------- импорт библиотек, контрольный ряд с известной закономерностью (синусойда + шум)
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

t = np.linspace(0, 80, 4000)
series = np.sin(t) + 0.01 * np.random.randn(len(t))
series = torch.tensor(series, dtype=torch.float32)

#print(t, series)
#--------------------------- sliding window, создание выборки тестовой и обучающей
WINDOW = 20
def make_windows(series, window):
    x, y = [], []
    for i in range(len(series) - window):
        x.append(series[i:i+window])
        y.append(series[i+window])
    return torch.stack(x), torch.stack(y)

X, y = make_windows(series, WINDOW)
split = int(len(X)*0.8)
X_train,y_train = X[:split],y[:split]
X_test,y_test = X[split:],y[split:]

#print("x train: ", X_train)
#print("y train: ", y_train)
#print("x test: ", X_test)
#print("y test: ", y_test)
#---------------------------  формирование датасета, объявление класса модели и создание объекта модели
X_train_lstm=X_train.unsqueeze(-1)
X_test_lstm=X_test.unsqueeze(-1)
lstm_loader=DataLoader(
    TensorDataset(X_train_lstm,y_train),
    batch_size=64,
    shuffle=True
)

class PriceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(input_size=1,hidden_size=32,batch_first=True)
        self.fc=nn.Linear(32,1)
    def forward(self,x):
        out,_=self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(1)

lstm_model=PriceLSTM()

#print(X_train_lstm.shape)
#--------------------------- обучение модели
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(lstm_model.parameters(),lr=0.001)

for epoch in range(30):
    lstm_model.train()
    for xb,yb in lstm_loader:
        optimizer.zero_grad()
        p=lstm_model(xb)
        loss=criterion(p,yb)
        loss.backward()
        optimizer.step()
    #print(epoch+1,loss.item())
#--------------------------- проверка точности модели
lstm_model.eval()
with torch.no_grad():
    pred=lstm_model(X_test_lstm)

lstm_mse=criterion(pred,y_test).item()
baseline=X_test[:,-1]
baseline_mse=criterion(baseline,y_test).item()

print("Baseline:",baseline_mse)
print("LSTM:",lstm_mse)
#--------------------------- графическое представление прогноза
plt.plot(y_test[:200].numpy(),label="real")
plt.plot(pred[:200].numpy(),label="LSTM")
plt.legend()
plt.show()
#--------------------------- рекурсивное предсказание
window=X_test[-1].clone()
future_lstm=[]
lstm_model.eval()
with torch.no_grad():
    for _ in range(500):
        x=window.reshape(1,WINDOW,1)
        nv=lstm_model(x)[0]
        future_lstm.append(nv.item())
        window=torch.cat([window[1:],nv.unsqueeze(0)])
#print(future_lstm[:10])
#--------------------------- 500 раз прогнозируем будущие значения
history = 500
plt.plot(
    range(len(series) - history, len(series)),
    series[-history:].numpy(),
    label="Known"
)
plt.plot(
    range(len(series), len(series) + len(future_lstm)),
    future_lstm,
    label="LSTM Forecast"
)
plt.legend()
plt.show()