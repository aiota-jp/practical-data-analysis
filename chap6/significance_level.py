import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)
alpha = 0.05

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 両側検定
z_crit = stats.norm.ppf(1 - alpha/2)
axes[0].plot(x, y, 'k-', linewidth=2)
axes[0].fill_between(x, y, where=(x <= -z_crit), alpha=0.4, color='red')
axes[0].fill_between(x, y, where=(x >= z_crit), alpha=0.4, color='red')
axes[0].axvline(-z_crit, color='red', linestyle='--')
axes[0].axvline(z_crit, color='red', linestyle='--')
axes[0].set_title(f"両側検定（α=5%: 各端2.5%）\n棄却域: |z| > {z_crit:.3f}", fontsize=12)
axes[0].set_xlabel("検定統計量")
axes[0].annotate("2.5%", xy=(-3.2, 0.01), fontsize=11, color='red')
axes[0].annotate("2.5%", xy=(2.8, 0.01), fontsize=11, color='red')
axes[0].annotate("95%（採択域）", xy=(-0.8, 0.15), fontsize=11)

# 片側検定（右側）
z_crit_one = stats.norm.ppf(1 - alpha)
axes[1].plot(x, y, 'k-', linewidth=2)
axes[1].fill_between(x, y, where=(x >= z_crit_one), alpha=0.4, color='red')
axes[1].axvline(z_crit_one, color='red', linestyle='--')
axes[1].set_title(f"片側検定・右側（α=5%）\n棄却域: z > {z_crit_one:.3f}", fontsize=12)
axes[1].set_xlabel("検定統計量")
axes[1].annotate("5%", xy=(2.2, 0.01), fontsize=11, color='red')
axes[1].annotate("95%（採択域）", xy=(-1.5, 0.15), fontsize=11)

plt.tight_layout()
plt.show()