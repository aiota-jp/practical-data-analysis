import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise
ts_nonstat = pd.Series(sales, index=dates, name="月次売上")

# グリッドサーチでAICが最小のパラメータを探索
best_aic = np.inf
best_order = None

for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            try:
                model = ARIMA(ts_nonstat, order=(p, d, q))
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
            except Exception:
                continue

print("=== 最適パラメータ（AIC最小） ===")
print(f"order: {best_order}")
print(f"AIC: {best_aic:.2f}")

# 最適モデルで再推定
best_model = ARIMA(ts_nonstat, order=best_order).fit()

print("\n=== 最適モデルのサマリー ===")
print(best_model.summary())