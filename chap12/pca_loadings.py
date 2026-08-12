import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# データの準備
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAの実行（全主成分を計算）
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# PCA結果
print("=== PCA結果 ===")
print(f"元の次元数: {X.shape[1]}")
print(f"固有値（分散）: {pca.explained_variance_}")
print(f"寄与率: {pca.explained_variance_ratio_}")
print(f"累積寄与率: {np.cumsum(pca.explained_variance_ratio_)}")

# 主成分負荷量
loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(X.shape[1])], index=iris.feature_names)

print("\n=== 主成分負荷量 ===")
print(loadings.round(4))

# ヒートマップで可視化
plt.figure(figsize=(8, 5))
sns.heatmap(loadings, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, center=0)
plt.title("主成分負荷量")
plt.xlabel("主成分")
plt.ylabel("特徴量")
plt.tight_layout()
plt.show()