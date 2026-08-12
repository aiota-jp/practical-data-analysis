import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import japanize_matplotlib

# Irisデータの準備
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_true = iris.target

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 階層的クラスタリング
agg = AgglomerativeClustering(
    n_clusters=3,          # クラスタ数
    metric='euclidean',    # 距離尺度
    linkage='ward'         # クラスタ間距離の計算方法
)
labels_agg = agg.fit_predict(X_scaled)

print("=== 階層的クラスタリング結果 ===")
print(f"クラスタ数: 3")
print(f"各クラスタのデータ数: {np.bincount(labels_agg)}")
