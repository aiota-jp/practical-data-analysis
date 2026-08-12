import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

# サンプルデータ（住宅データ）
np.random.seed(42)
n = 100
area = np.random.uniform(40, 120, n)       # 面積（㎡）
distance = np.random.uniform(1, 30, n)     # 駅距離（分）
age = np.random.uniform(0, 40, n)          # 築年数
rooms = np.random.randint(1, 5, n)         # 部屋数

# DataFrameの作成
df = pd.DataFrame({"area": area, "distance": distance, "age": age, "rooms": rooms})

# 相関行列の計算
corr_matrix = df[["area", "distance", "age", "rooms"]].corr()

print("=== 相関行列 ===")
print(corr_matrix.round(3))

# ヒートマップで可視化
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, square=True)
plt.title("説明変数間の相関行列（ヒートマップ）")
plt.tight_layout()
plt.show()