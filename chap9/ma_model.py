import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# MA(2)モデルのサンプルデータを生成
np.random.seed(42)
n = 200
eps = np.random.normal(0, 1, n)
y_ma = np.zeros(n)

for t in range(2, n):
    y_ma[t] = eps[t] + 0.7 * eps[t-1] + 0.3 * eps[t-2]

ts_ma = pd.Series(y_ma, name="MA(2)データ")

# MA(2)モデルの推定
model_ma = ARIMA(ts_ma, order=(0, 0, 2))
result_ma = model_ma.fit()

print("=== MA(2)モデルの推定結果 ===")
print(f"θ1 = {result_ma.params['ma.L1']:.4f}")
print(f"θ2 = {result_ma.params['ma.L2']:.4f}")
print(f"AIC = {result_ma.aic:.2f}")