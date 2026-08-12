import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Irisデータの準備
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-meansクラスタリング
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)

# クラスタリング結果
print("=== k-meansクラスタリング結果 ===")
print(f"クラスタ数: {kmeans.n_clusters}")
print(f"各クラスタのデータ数: {np.bincount(labels_km)}")
print(f"SSE（慣性）: {kmeans.inertia_:.2f}")

# クラスタ中心
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=iris.feature_names)

print("\n=== クラスタ中心（元のスケール） ===")
print(centers.round(2))