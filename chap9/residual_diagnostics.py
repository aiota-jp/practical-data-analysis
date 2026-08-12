import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 季節性のある時系列データの生成
np.random.seed(42)
n = 96
dates = pd.date_range(start="2017-01-01", periods=n, freq="MS")
trend = np.linspace(100, 180, n)
seasonal = 25 * np.sin(2 * np.pi * np.arange(n) / 12) + 15 * np.cos(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 8, n)
ts_seasonal = pd.Series(trend + seasonal + noise, index=dates, name="月次売上")

# 学習データ
train = ts_seasonal[:"2023-12"]

# SARIMAモデルの推定
model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
result_sarima = model_sarima.fit(disp=False)

# 残差診断
residuals = result_sarima.resid.dropna()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 残差の推移
axes[0,0].plot(residuals)
axes[0,0].set_title("残差の推移")
axes[0,0].axhline(0, color="red", linestyle="--")
axes[0,0].grid(alpha=0.3)

# 残差のヒストグラム
axes[0,1].hist(residuals, bins=20, edgecolor="black", alpha=0.7, density=True)
axes[0,1].set_title("残差のヒストグラム")

# 残差のACF
plot_acf(residuals, lags=24, ax=axes[1,0])
axes[1,0].set_title("残差のACF")

# Q-Qプロット
stats.probplot(residuals, plot=axes[1,1])
axes[1,1].set_title("残差のQ-Qプロット")

plt.tight_layout()
plt.show()

# Ljung-Box検定
lb_result = acorr_ljungbox(residuals, lags=[12], return_df=True)
print("\n=== Ljung-Box検定 ===")
print(lb_result)

if lb_result["lb_pvalue"].iloc[0] > 0.05:
    print("→ 残差に有意な自己相関は確認されませんでした。")
else:
    print("→ 残差に有意な自己相関が確認されました。モデルの改善を検討します。")