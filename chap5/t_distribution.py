import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# t分布と正規分布の比較
x = np.linspace(-4, 4, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, stats.norm.pdf(x), 'k-', linewidth=2, label='正規分布 N(0,1)')
for df in [1, 3, 10, 30]:
    plt.plot(x, stats.t.pdf(x, df), '--', linewidth=1.5, label=f't分布 (自由度={df})')
plt.xlabel("x", fontsize=12)
plt.ylabel("確率密度", fontsize=12)
plt.title("t分布と正規分布の比較", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()