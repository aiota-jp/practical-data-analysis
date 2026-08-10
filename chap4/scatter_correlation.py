import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("相関係数と散布図の関係", fontsize=16)

# さまざまな相関を持つデータを生成
correlations = [1.0, 0.8, 0.4, 0.0, -0.6, -0.9]
titles = ["R≈1.0（完全な正の相関）", "R≈0.8（強い正の相関）", "R≈0.4（中程度の相関）",
          "R≈0.0（無相関）", "R≈-0.6（負の相関）", "R≈-0.9（強い負の相関）"]

for idx, (r, title) in enumerate(zip(correlations, titles)):
    ax = axes[idx // 3, idx % 3]
    
    # 相関のあるデータを生成
    mean = [0, 0]
    cov = [[1, r], [r, 1]]
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    
    ax.scatter(x, y, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.show()