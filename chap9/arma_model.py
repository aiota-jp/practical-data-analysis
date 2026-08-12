import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# AR(2)のサンプルデータを生成
np.random.seed(42)
n = 200
y = np.zeros(n)

for t in range(2, n):
    y[t] = 0.6 * y[t-1] + 0.2 * y[t-2] + np.random.normal(0, 1)

ts_ar = pd.Series(y, name="AR(2)データ")

# ARMA(2,1)モデルの推定
# ARIMAのorder=(p, 0, q)として指定
model_arma = ARIMA(ts_ar, order=(2, 0, 1))
result_arma = model_arma.fit()

print("=== ARMA(2,1)モデルの推定結果 ===")
print(f"AR係数: {result_arma.arparams}")
print(f"MA係数: {result_arma.maparams}")
print(f"AIC: {result_arma.aic:.2f}")
print(f"BIC: {result_arma.bic:.2f}")