import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import japanize_matplotlib

# データの準備
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 標準化（PCAの前に必須）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAの実行（全主成分を計算）
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("=== PCA結果 ===")
print(f"元の次元数: {X.shape[1]}")
print(f"固有値（分散）: {pca.explained_variance_}")
print(f"寄与率: {pca.explained_variance_ratio_}")
print(f"累積寄与率: {np.cumsum(pca.explained_variance_ratio_)}")
