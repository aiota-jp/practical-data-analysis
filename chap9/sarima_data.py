import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import japanize_matplotlib

# 季節性のある時系列データの生成
np.random.seed(42)
n = 96  # 8年分の月次データ
dates = pd.date_range(start='2017-01-01', periods=n, freq='MS')

trend = np.linspace(100, 180, n)
seasonal = 25 * np.sin(2 * np.pi * np.arange(n) / 12) + 15 * np.cos(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 8, n)

ts_seasonal = pd.Series(trend + seasonal + noise, index=dates, name="月次売上")

# データの可視化
plt.figure(figsize=(12, 5))
plt.plot(ts_seasonal, color='steelblue', linewidth=1.5)
plt.xlabel("日付")
plt.ylabel("売上")
plt.title("季節性のある月次売上データ")
plt.grid(alpha=0.3)
plt.show()

# 学習データとテストデータの分割
train = ts_seasonal[:'2023-12']
test = ts_seasonal['2024-01':]

print(f"学習データ: {train.index[0].strftime('%Y-%m')} 〜 {train.index[-1].strftime('%Y-%m')} ({len(train)}件)")
print(f"テストデータ: {test.index[0].strftime('%Y-%m')} 〜 {test.index[-1].strftime('%Y-%m')} ({len(test)}件)")