import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import japanize_matplotlib

# 複数変数のサンプルデータ
np.random.seed(42)
n = 100
df = pd.DataFrame({
    "気温": np.random.normal(25, 5, n),
    "湿度": np.random.normal(60, 10, n),
    "ビール売上": None,
    "アイス売上": None,
    "コート売上": None,
})
df["ビール売上"] = 50 + 8 * df["気温"] + np.random.normal(0, 20, n)
df["アイス売上"] = 30 + 10 * df["気温"] + np.random.normal(0, 15, n)
df["コート売上"] = 500 - 12 * df["気温"] + np.random.normal(0, 25, n)

# 相関行列のヒートマップ
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
            vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("相関行列のヒートマップ", fontsize=14)
plt.show()