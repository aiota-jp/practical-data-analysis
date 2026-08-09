import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import japanize_matplotlib

x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'k-', linewidth=2)

# 各範囲を塗りつぶし
plt.fill_between(x, y, where=(x >= -3) & (x <= 3), alpha=0.1, color='blue', label='±3σ (99.7%)')
plt.fill_between(x, y, where=(x >= -2) & (x <= 2), alpha=0.2, color='blue', label='±2σ (95.4%)')
plt.fill_between(x, y, where=(x >= -1) & (x <= 1), alpha=0.3, color='blue', label='±1σ (68.3%)')

plt.xlabel("Zスコア", fontsize=12)
plt.ylabel("確率密度", fontsize=12)
plt.title("標準正規分布と68-95-99.7ルール", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()
