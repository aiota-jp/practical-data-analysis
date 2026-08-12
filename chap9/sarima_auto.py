import numpy as np
import pandas as pd
from pmdarima import auto_arima

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

# 季節性ありのauto_arima
auto_model = auto_arima(train, start_p=0, max_p=3, start_q=0, max_q=3, d=None, start_P=0, max_P=2, start_Q=0, max_Q=2, D=None, seasonal=True, m=12, stepwise=True, trace=True)

print("=== auto_arima（季節性あり）の結果 ===")
print(f"order: {auto_model.order}")
print(f"seasonal_order: {auto_model.seasonal_order}")
print(f"AIC: {auto_model.aic():.2f}")
print(auto_model.summary())

# テスト期間を予測
forecast_auto, conf_int = auto_model.predict(n_periods=len(test), return_conf_int=True)

print("\n=== 予測値 ===")
print(forecast_auto)