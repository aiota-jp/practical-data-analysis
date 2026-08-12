import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# t分布とp値の可視化
df = 29  # 自由度（n-1）
t_observed = -2.5  # 観測されたt値

x = np.linspace(-4, 4, 1000)
y = stats.t.pdf(x, df)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'k-', linewidth=2, label='t分布 (df=29)')

# 両側検定のp値（棄却域）を塗りつぶし
plt.fill_between(x, y, where=(x <= t_observed), alpha=0.4, color='red',
                 label=f'p値の領域 (t≤{t_observed})')
plt.fill_between(x, y, where=(x >= -t_observed), alpha=0.4, color='red',
                 label=f'p値の領域 (t≥{-t_observed})')

# p値を計算
p_value = 2 * stats.t.cdf(t_observed, df)

plt.axvline(t_observed, color='red', linestyle='--', linewidth=1.5)
plt.axvline(-t_observed, color='red', linestyle='--', linewidth=1.5)

plt.xlabel("t値", fontsize=12)
plt.ylabel("確率密度", fontsize=12)
plt.title(f"t分布とp値の関係（t={t_observed}, p={p_value:.4f}）", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()