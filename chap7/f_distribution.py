import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

x = np.linspace(0, 5, 200)

plt.figure(figsize=(10, 6))
for dfn, dfd in [(2, 27), (3, 36), (5, 50), (10, 100)]:
    y = stats.f.pdf(x, dfn, dfd)
    plt.plot(x, y, linewidth=2, label=f'F({dfn}, {dfd})')

plt.xlabel("F値", fontsize=12)
plt.ylabel("確率密度", fontsize=12)
plt.title("F分布（自由度別）", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xlim(0, 5)
plt.show()