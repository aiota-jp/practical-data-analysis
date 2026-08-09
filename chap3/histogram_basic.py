import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示対応

# サンプルデータの生成（正規分布に従うテストスコア）
np.random.seed(42)
scores = np.random.normal(loc=65, scale=15, size=200)

# 基本的なヒストグラム
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel("テストスコア", fontsize=12)
plt.ylabel("度数（人数）", fontsize=12)
plt.title("テストスコアの分布", fontsize=14)
plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'平均: {np.mean(scores):.1f}')
plt.axvline(np.median(scores), color='green', linestyle='--', label=f'中央値: {np.median(scores):.1f}')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.show()
