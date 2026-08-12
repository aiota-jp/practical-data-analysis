import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import japanize_matplotlib

# トレンドのある非定常データの生成
np.random.seed(42)
n = 120
trend = np.linspace(100, 250, n)
noise = np.random.normal(0, 10, n)

ts_nonstat = pd.Series(trend + noise,
                       index=pd.date_range('2015-01', periods=n, freq='MS'),
                       name="売上")

# 定常性の確認
result = adfuller(ts_nonstat)
print(f"=== 元データのADF検定 ===")
print(f"p値: {result[1]:.6f} → {'定常' if result[1] < 0.05 else '非定常'}")

# 1次差分後の定常性
ts_diff = ts_nonstat.diff().dropna()
result_diff = adfuller(ts_diff)
print(f"\n=== 1次差分後のADF検定 ===")
print(f"p値: {result_diff[1]:.6f} → {'定常' if result_diff[1] < 0.05 else '非定常'}")

# ARIMA(1,1,1)モデルの構築
model = ARIMA(ts_nonstat, order=(1, 1, 1))
result_arima = model.fit()

print(f"\n=== ARIMA(1,1,1)モデル ===")
print(f"AIC: {result_arima.aic:.2f}")
print(f"BIC: {result_arima.bic:.2f}")
print(result_arima.summary())