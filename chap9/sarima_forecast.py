import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 季節性のある時系列データの生成
np.random.seed(42)
n = 96
dates = pd.date_range(start="2017-01-01", periods=n, freq="MS")
trend = np.linspace(100, 180, n)
seasonal = 25 * np.sin(2 * np.pi * np.arange(n) / 12) + 15 * np.cos(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 8, n)
ts_seasonal = pd.Series(trend + seasonal + noise, index=dates, name="月次売上")

# 学習データとテストデータの分割
train = ts_seasonal[:"2023-12"]
test = ts_seasonal["2024-01":]

# SARIMAモデルの推定
model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
result_sarima = model_sarima.fit(disp=False)

# テスト期間の予測
forecast = result_sarima.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()


# 予測精度の評価
rmse = np.sqrt(mean_squared_error(test, forecast_mean))
mae = mean_absolute_error(test, forecast_mean)
mape = np.mean(np.abs((test - forecast_mean) / test)) * 100

print("=== 予測精度 ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# 可視化
plt.figure(figsize=(14, 6))
plt.plot(train[-24:], label="学習データ", color="steelblue", linewidth=1.5)
plt.plot(test, label="実測値（テスト）", color="green", linewidth=1.5, linestyle="--")
plt.plot(forecast_mean, label="予測値", color="red", linewidth=2)
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.2, color="red", label="95%予測区間")
plt.xlabel("日付")
plt.ylabel("売上")
plt.title("SARIMAモデルによる売上予測")
plt.legend()
plt.grid(alpha=0.3)
plt.show()