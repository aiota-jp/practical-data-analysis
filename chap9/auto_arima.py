import numpy as np
import pandas as pd
from pmdarima import auto_arima

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise
ts_nonstat = pd.Series(sales, index=dates, name="月次売上")

# auto_arimaによるパラメータの自動選択
auto_model = auto_arima(ts_nonstat, start_p=0, max_p=5, start_q=0, max_q=5, d=None, seasonal=False, stepwise=True, trace=True)

print("\n=== auto_arima 結果 ===")
print(f"最適なorder: {auto_model.order}")
print(f"AIC: {auto_model.aic():.2f}")
print(auto_model.summary())