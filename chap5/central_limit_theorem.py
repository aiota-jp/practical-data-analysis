import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

np.random.seed(42)

# さまざまな母集団分布で中心極限定理を確認
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

distributions = {
    "一様分布": np.random.uniform(0, 100, 10000),
    "指数分布": np.random.exponential(50, 10000),
    "二項分布": np.random.binomial(10, 0.3, 10000),
}

sample_sizes = [5, 30, 100]

for row, (dist_name, population) in enumerate(distributions.items()):
    for col, n in enumerate(sample_sizes):
        means = [np.random.choice(population, n).mean() for _ in range(1000)]
        axes[row, col].hist(means, bins=30, edgecolor='black', alpha=0.7,
                           density=True, color='steelblue')
        axes[row, col].set_title(f"{dist_name}\nn={n}", fontsize=10)
        if col == 0:
            axes[row, col].set_ylabel("確率密度")

fig.suptitle("中心極限定理: 標本サイズが大きくなると標本平均の分布は正規分布に近づく",
             fontsize=13)
plt.tight_layout()
plt.show()