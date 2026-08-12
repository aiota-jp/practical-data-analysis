import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# AR(2)のサンプルデータを生成
np.random.seed(42)
n = 200
y = np.zeros(n)

for t in range(2, n):
    y[t] = 0.6 * y[t-1] + 0.2 * y[t-2] + np.random.normal(0, 1)

ts_ar = pd.Series(y, name="AR(2)データ")

# 複数のARMAモデルを比較
results = []

for p in range(0, 4):
    for q in range(0, 4):
        try:
            model = ARIMA(ts_ar, order=(p, 0, q))
            fit = model.fit()
            results.append({"p": p, "q": q, "AIC": fit.aic, "BIC": fit.bic})
        except Exception:
            continue

# AICの小さい順に並べる
df_results = pd.DataFrame(results).sort_values("AIC")

print("=== AICによるモデル比較（上位5件） ===")
print(df_results.head().round(2).to_string(index=False))

# 最良モデル
best = df_results.iloc[0]
print(f"\n最良モデル: ARMA({int(best['p'])},{int(best['q'])})")
print(f"AIC: {best['AIC']:.2f}")
print(f"BIC: {best['BIC']:.2f}")