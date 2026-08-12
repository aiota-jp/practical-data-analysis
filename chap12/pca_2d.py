from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Irisデータの準備
iris = load_iris()
X = iris.data

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2次元に削減
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# 累積寄与率
cum_ratio = pca_2d.explained_variance_ratio_.sum()

print("=== 2次元PCA ===")
print(f"第1主成分の寄与率: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"第2主成分の寄与率: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"累積寄与率: {cum_ratio:.4f}")
print(f"元データの情報の約{cum_ratio * 100:.1f}%を保持")