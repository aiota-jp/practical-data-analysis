import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import japanize_matplotlib

# 正規分布の描画
x = np.linspace(-4, 4, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：平均が異なる正規分布
for mu in [-1, 0, 1]:
    y = stats.norm.pdf(x, loc=mu, scale=1)
    axes[0].plot(x, y, label=f'μ={mu}, σ=1')
axes[0].set_title("平均が異なる正規分布", fontsize=13)
axes[0].set_xlabel("x")
axes[0].set_ylabel("確率密度")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右：標準偏差が異なる正規分布
for sigma in [0.5, 1, 2]:
    y = stats.norm.pdf(x, loc=0, scale=sigma)
    axes[1].plot(x, y, label=f'μ=0, σ={sigma}')
axes[1].set_title("標準偏差が異なる正規分布", fontsize=13)
axes[1].set_xlabel("x")
axes[1].set_ylabel("確率密度")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
