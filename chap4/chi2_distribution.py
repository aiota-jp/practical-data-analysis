from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

x = np.linspace(0, 15, 200)

plt.figure(figsize=(10, 6))
for k in [1, 2, 3, 5, 7]:
    y = stats.chi2.pdf(x, df=k)
    plt.plot(x, y, label=f'自由度 k={k}', linewidth=2)

plt.xlabel("カイ二乗値", fontsize=12)
plt.ylabel("確率密度", fontsize=12)
plt.title("カイ二乗分布（自由度別）", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.ylim(0, 0.5)
plt.show()