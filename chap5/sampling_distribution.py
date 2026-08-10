import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

np.random.seed(42)

# 母集団（平均500、標準偏差100）
population = np.random.normal(loc=500, scale=100, size=10000)

# 標本平均を1000回繰り返し計算
n_trials = 1000
sample_size = 50
sample_means = []

for _ in range(n_trials):
    sample = np.random.choice(population, size=sample_size, replace=False)
    sample_means.append(sample.mean())

sample_means = np.array(sample_means)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 母集団の分布
axes[0].hist(population, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
axes[0].axvline(population.mean(), color='red', linestyle='--', label=f'母平均: {population.mean():.1f}')
axes[0].set_title("母集団の分布", fontsize=13)
axes[0].set_xlabel("年収（万円）")
axes[0].set_ylabel("度数")
axes[0].legend()

# 標本平均の分布（標本分布）
axes[1].hist(sample_means, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
axes[1].axvline(sample_means.mean(), color='red', linestyle='--', 
                label=f'標本平均の平均: {sample_means.mean():.1f}')
axes[1].set_title(f"標本平均の分布（n={sample_size}, 試行{n_trials}回）", fontsize=13)
axes[1].set_xlabel("標本平均（万円）")
axes[1].set_ylabel("度数")
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"標本平均の平均: {sample_means.mean():.2f}")
print(f"標本平均の標準偏差（標準誤差）: {sample_means.std():.2f}")
print(f"理論的な標準誤差: {population.std() / np.sqrt(sample_size):.2f}")