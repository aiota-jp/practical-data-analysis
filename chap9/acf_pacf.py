import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise

# 時系列データの作成
ts = pd.Series(sales, index=dates, name="月次売上")

# ACF・PACFの可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# 自己相関関数（ACF）
plot_acf(ts, lags=24, ax=axes[0])
axes[0].set_title("自己相関関数（ACF）")

# 偏自己相関関数（PACF）
plot_pacf(ts, lags=24, ax=axes[1], method="ywm")
axes[1].set_title("偏自己相関関数（PACF）")

plt.tight_layout()
plt.show()