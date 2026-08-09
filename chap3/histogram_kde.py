import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示対応
import seaborn as sns

# サンプルデータの生成（正規分布に従うテストスコア）
np.random.seed(42)
scores = np.random.normal(loc=65, scale=15, size=200)

plt.figure(figsize=(10, 6))
sns.histplot(scores, bins=15, kde=True, color='steelblue', edgecolor='black')
plt.xlabel("テストスコア", fontsize=12)
plt.ylabel("度数", fontsize=12)
plt.title("ヒストグラム + カーネル密度推定（KDE）", fontsize=14)
plt.show()
