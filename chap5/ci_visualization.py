import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

np.random.seed(42)

# 母集団
population = np.random.normal(loc=50, scale=10, size=100000)
true_mean = population.mean()

# 20回の標本抽出で信頼区間を計算
n_experiments = 20
sample_size = 30

plt.figure(figsize=(10, 8))
contains_true = 0

for i in range(n_experiments):
    sample = np.random.choice(population, size=sample_size, replace=False)
    x_bar = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(sample_size)
    ci = stats.t.interval(0.95, sample_size-1, loc=x_bar, scale=se)
    
    # 真の母平均を含むかどうかで色を変える
    if ci[0] <= true_mean <= ci[1]:
        color = 'steelblue'
        contains_true += 1
    else:
        color = 'red'
    
    plt.plot([ci[0], ci[1]], [i, i], color=color, linewidth=2)
    plt.plot(x_bar, i, 'o', color=color, markersize=6)

plt.axvline(true_mean, color='green', linestyle='--', linewidth=2, 
            label=f'真の母平均: {true_mean:.2f}')
plt.xlabel("値", fontsize=12)
plt.ylabel("実験番号", fontsize=12)
plt.title(f"95%信頼区間の繰り返し実験（{n_experiments}回中{contains_true}回が母平均を含む）",
          fontsize=13)
plt.legend(fontsize=11)
plt.grid(alpha=0.3, axis='x')
plt.show()