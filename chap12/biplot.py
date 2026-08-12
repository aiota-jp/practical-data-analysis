import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Irisデータの準備
iris = load_iris()
X = iris.data
y = iris.target

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAで2次元に削減
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# バイプロット
fig, ax = plt.subplots(figsize=(9, 7))

# データ点のプロット
for i, name in enumerate(iris.target_names):
    mask = y == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=50, alpha=0.6, edgecolors="black", linewidths=0.5, label=name)

# 特徴量の寄与方向を矢印で表示
scale = 3
for j, feature in enumerate(iris.feature_names):
    x_arrow = pca_2d.components_[0, j] * scale
    y_arrow = pca_2d.components_[1, j] * scale
    ax.arrow(0, 0, x_arrow, y_arrow, head_width=0.08, head_length=0.08, fc="red", ec="red", alpha=0.8)
    ax.text(x_arrow * 1.15, y_arrow * 1.15, feature, color="red", fontsize=9, ha="center")

ax.set_xlabel("第1主成分")
ax.set_ylabel("第2主成分")
ax.set_title("バイプロット（データ点 + 特徴量の寄与方向）")
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
plt.tight_layout()
plt.show()