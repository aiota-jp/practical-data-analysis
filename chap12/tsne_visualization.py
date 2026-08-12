import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Irisデータの準備
iris = load_iris()
X = iris.data
y = iris.target

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAで2次元に変換
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# t-SNEで2次元に変換
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# PCAとt-SNEを比較
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA
for i, name in enumerate(iris.target_names):
    mask = y == i
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], s=50, alpha=0.7, edgecolors="black", linewidths=0.5, label=name)

axes[0].set_title("PCA")
axes[0].set_xlabel("第1主成分")
axes[0].set_ylabel("第2主成分")
axes[0].legend()
axes[0].grid(alpha=0.3)

# t-SNE
for i, name in enumerate(iris.target_names):
    mask = y == i
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=50, alpha=0.7, edgecolors="black", linewidths=0.5, label=name)

axes[1].set_title("t-SNE")
axes[1].set_xlabel("t-SNE 次元1")
axes[1].set_ylabel("t-SNE 次元2")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()