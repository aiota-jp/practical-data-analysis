import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示対応

# サンプルデータの生成（正規分布に従うテストスコア）
np.random.seed(42)
scores = np.random.normal(loc=65, scale=15, size=200)
n = len(scores)

# スタージェスの公式
k = int(np.log2(n) + 1)
bin_width = (scores.max() - scores.min()) / k

print(f"データ数: {n}")
print(f"推奨階級数: {k}")
print(f"階級幅: {bin_width:.1f}")

# スタージェスの公式によるビン数でヒストグラム作成
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=k, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel("テストスコア", fontsize=12)
plt.ylabel("度数", fontsize=12)
plt.title(f"ヒストグラム（スタージェスの公式: bins={k}）", fontsize=14)
plt.show()
