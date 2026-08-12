import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Irisデータの準備
iris = load_iris()
X = iris.data
y_true = iris.target

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-meansクラスタリング
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)

# PCAで2次元に削減
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# k-meansの結果
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap="Set2", s=50, alpha=0.7, edgecolors="black")
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", marker="X", s=200, edgecolors="black", linewidths=2, label="セントロイド")
axes[0].set_title("k-meansクラスタリング結果")
axes[0].set_xlabel("第1主成分")
axes[0].set_ylabel("第2主成分")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 正解ラベルとの比較
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="Set2", s=50, alpha=0.7, edgecolors="black")
axes[1].set_title("正解ラベル（参考）")
axes[1].set_xlabel("第1主成分")
axes[1].set_ylabel("第2主成分")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()