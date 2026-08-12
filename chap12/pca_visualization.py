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

# PCA結果の可視化
plt.figure(figsize=(8, 6))

for i, name in enumerate(iris.target_names):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=60, alpha=0.7, edgecolors="black", linewidths=0.5, label=name)

plt.xlabel(f"第1主成分（寄与率: {pca_2d.explained_variance_ratio_[0]:.3f}）")
plt.ylabel(f"第2主成分（寄与率: {pca_2d.explained_variance_ratio_[1]:.3f}）")
plt.title("PCAによるIrisデータの2次元可視化")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()