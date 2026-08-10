# アンスコムの4つのデータセット — 相関係数が同じでも散布図は全く異なる
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

anscombe = sns.load_dataset("anscombe")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for idx, dataset in enumerate(["I", "II", "III", "IV"]):
    ax = axes[idx // 2, idx % 2]
    subset = anscombe[anscombe["dataset"] == dataset]
    r = subset["x"].corr(subset["y"])
    ax.scatter(subset["x"], subset["y"], s=50, edgecolors='black')
    ax.set_title(f"Dataset {dataset} (R={r:.3f})", fontsize=11)
    ax.set_xlim(2, 20)
    ax.set_ylim(2, 14)

plt.suptitle("アンスコムの4つのデータセット\n相関係数はほぼ同じだが散布図は全く異なる", fontsize=13)
plt.tight_layout()
plt.show()