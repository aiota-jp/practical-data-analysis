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

# 階層的クラスタリング（ウォード法）
linked = linkage(X_scaled, method="ward")

# デンドログラムの表示
plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9, color_threshold=7)
plt.xlabel("サンプル")
plt.ylabel("距離")
plt.title("デンドログラム（ウォード法）")
plt.axhline(y=7, color="red", linestyle="--", label="切断位置")
plt.legend()
plt.tight_layout()
plt.show()