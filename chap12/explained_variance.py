import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# データの準備
iris = load_iris()
X = iris.data

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAの実行
pca = PCA()
pca.fit(X_scaled)

# 主成分数
components = range(1, len(pca.explained_variance_ratio_) + 1)

# 累積寄与率
cum_ratio = np.cumsum(pca.explained_variance_ratio_)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 寄与率の棒グラフ
axes[0].bar(components, pca.explained_variance_ratio_, color="steelblue", alpha=0.8, edgecolor="black")
axes[0].set_xlabel("主成分")
axes[0].set_ylabel("寄与率")
axes[0].set_title("各主成分の寄与率")
axes[0].set_xticks(components)
axes[0].grid(axis="y", alpha=0.3)

for i, v in enumerate(pca.explained_variance_ratio_):
    axes[0].text(i + 1, v + 0.01, f"{v:.3f}", ha="center")

# 累積寄与率の折れ線グラフ
axes[1].plot(components, cum_ratio, "o-", color="steelblue", markersize=10, linewidth=2)
axes[1].axhline(0.8, color="red", linestyle="--", alpha=0.7, label="80%ライン")
axes[1].axhline(0.9, color="orange", linestyle="--", alpha=0.7, label="90%ライン")
axes[1].set_xlabel("主成分数")
axes[1].set_ylabel("累積寄与率")
axes[1].set_title("累積寄与率")
axes[1].set_xticks(components)
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].grid(alpha=0.3)

for i, v in enumerate(cum_ratio):
    axes[1].text(i + 1, v + 0.02, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.show()

# 必要な主成分数を確認
print("=== 累積寄与率による主成分数の決定 ===")
for threshold in [0.8, 0.9, 0.95]:
    n_components = np.argmax(cum_ratio >= threshold) + 1
    print(f"累積寄与率{threshold * 100:.0f}%以上 → {n_components}次元")