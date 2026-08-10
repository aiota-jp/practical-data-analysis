import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# 二項分布の可視化
n_trials = 100
p = 0.3

x = np.arange(0, n_trials + 1)
pmf = stats.binom.pmf(x, n_trials, p)

plt.figure(figsize=(10, 6))
plt.bar(x, pmf, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
plt.axvline(n_trials * p, color='red', linestyle='--', linewidth=2, 
            label=f'期待値 np = {n_trials * p}')
plt.xlabel("事象が起こった回数 k", fontsize=12)
plt.ylabel("確率 P(X=k)", fontsize=12)
plt.title(f"二項分布 Bin(n={n_trials}, p={p})", fontsize=14)
plt.legend(fontsize=11)
plt.xlim(10, 50)
plt.grid(alpha=0.3)
plt.show()

print(f"平均: np = {n_trials * p}")
print(f"分散: np(1-p) = {n_trials * p * (1-p)}")
print(f"標準偏差: √np(1-p) = {np.sqrt(n_trials * p * (1-p)):.2f}")