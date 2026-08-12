import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Irisデータの準備
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# リンク方法ごとのデンドログラムを比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
methods = ["ward", "complete", "average", "single"]

for ax, method in zip(axes.ravel(), methods):
    linked = linkage(X_scaled, method=method)
    dendrogram(linked, ax=ax, truncate_mode="lastp", p=20, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f"リンク方法: {method}", fontsize=12)
    ax.set_xlabel("クラスタ")
    ax.set_ylabel("距離")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()