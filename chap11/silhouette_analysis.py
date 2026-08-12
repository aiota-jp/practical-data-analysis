import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Irisデータの準備
iris = load_iris()
X = iris.data

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 各クラスタ数でシルエットスコアを計算
k_range = range(2, 11)
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

# 最適なクラスタ数
best_k = list(k_range)[np.argmax(sil_scores)]

# 可視化
plt.figure(figsize=(8, 5))
plt.plot(k_range, sil_scores, "o-", color="steelblue", markersize=8, linewidth=2)
plt.xlabel("クラスタ数 K")
plt.ylabel("シルエットスコア（平均）")
plt.title("シルエット分析によるクラスタ数の決定")
plt.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"最大スコア（K={best_k}）")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(k_range)
plt.show()

print(f"シルエットスコアが最大のクラスタ数: K={best_k}（スコア={max(sil_scores):.4f}）")