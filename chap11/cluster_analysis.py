import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
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
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)

# クラスタラベルを元データに付与
df_result = X.copy()
df_result["cluster"] = labels_km

# クラスタごとの平均値
print("=== クラスタごとの平均値 ===")
print(df_result.groupby("cluster").mean().round(2))

# クラスタごとの特徴を可視化
for col in iris.feature_names:
    plt.figure(figsize=(8, 3))
    sns.boxplot(x="cluster", y=col, data=df_result, hue="cluster", palette="Set2", legend=False)
    plt.xlabel("クラスタ")
    plt.ylabel(col)
    plt.title(f"クラスタ別：{col}")
    plt.tight_layout()
    plt.show()