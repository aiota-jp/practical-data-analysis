import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import japanize_matplotlib

# サンプルデータ
np.random.seed(42)
df = pd.DataFrame({
    "年齢": np.random.randint(20, 60, 100),
    "年収": np.random.randint(300, 1200, 100),
    "勤続年数": np.random.randint(1, 30, 100),
    "満足度": np.random.randint(1, 10, 100)
})

# 相関行列
corr_matrix = df.corr()

# ヒートマップ
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("相関行列のヒートマップ", fontsize=14)
plt.show()
