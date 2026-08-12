import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import japanize_matplotlib

# AR(2)モデルのデータを生成
np.random.seed(42)
n = 200
y = np.zeros(n)
y[0], y[1] = 0, 0
for t in range(2, n):
    y[t] = 0.6 * y[t-1] + 0.2 * y[t-2] + np.random.normal(0, 1)

ts_ar = pd.Series(y, name="AR(2)データ")

# ARモデルの推定（order=(p, d, q) = (2, 0, 0)）
model_ar = ARIMA(ts_ar, order=(2, 0, 0))
result_ar = model_ar.fit()

print("=== AR(2)モデルの推定結果 ===")
print(result_ar.summary())

# 予測
forecast = result_ar.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# 可視化
plt.figure(figsize=(12, 5))
plt.plot(ts_ar[-50:], label='実測値', color='steelblue')
plt.plot(range(200, 230), forecast_mean, label='予測', color='red', linewidth=2)
plt.fill_between(range(200, 230), forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                 alpha=0.2, color='red', label='95%信頼区間')
plt.xlabel("時点")
plt.ylabel("値")
plt.title("AR(2)モデルによる予測")
plt.legend()
plt.grid(alpha=0.3)
plt.show()