import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import japanize_matplotlib

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=72, freq='MS')
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise

ts = pd.Series(sales, index=dates, name="月次売上")

# 基本統計量
print("=== 時系列データの基本情報 ===")
print(f"期間: {ts.index[0].strftime('%Y-%m')} 〜 {ts.index[-1].strftime('%Y-%m')}")
print(f"データ数: {len(ts)}")
print(f"平均: {ts.mean():.2f}")
print(f"標準偏差: {ts.std():.2f}")

# 時系列プロット
plt.figure(figsize=(12, 4))
plt.plot(ts, color='steelblue', linewidth=1.5)
plt.xlabel("日付")
plt.ylabel("売上")
plt.title("月次売上データの推移")
plt.grid(alpha=0.3)
plt.show()