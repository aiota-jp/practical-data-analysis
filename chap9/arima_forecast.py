import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.tsa.arima.model import ARIMA

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise
ts_nonstat = pd.Series(sales, index=dates, name="月次売上")

# ARIMA(1,1,1)モデル
model_arima = ARIMA(ts_nonstat, order=(1, 1, 1))
result_arima = model_arima.fit()

# 12ヶ月先まで予測
forecast = result_arima.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# 可視化
plt.figure(figsize=(12, 5))
plt.plot(ts_nonstat, label="実測値", color="steelblue", linewidth=1.5)
plt.plot(forecast_mean, label="予測値", color="red", linewidth=2)
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.2, color="red", label="95%予測区間")
plt.xlabel("日付")
plt.ylabel("売上")
plt.title("ARIMA(1,1,1)モデルによる売上予測")
plt.legend()
plt.grid(alpha=0.3)
plt.show()