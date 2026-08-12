import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Irisデータの準備
iris = load_iris()
X = iris.data

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K=2, 3, 4のシルエット分析
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for idx, k in enumerate([2, 3, 4]):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_vals = silhouette_samples(X_scaled, labels)

    y_lower = 10
    for i in range(k):
        cluster_sil = np.sort(sil_vals[labels == i])
        y_upper = y_lower + len(cluster_sil)
        color = cm.Set2(i / k)
        axes[idx].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, facecolor=color, alpha=0.7)
        axes[idx].text(-0.05, y_lower + 0.5 * len(cluster_sil), str(i))
        y_lower = y_upper + 10

    avg_score = silhouette_score(X_scaled, labels)
    axes[idx].axvline(avg_score, color="red", linestyle="--", label=f"平均={avg_score:.3f}")
    axes[idx].set_title(f"K={k}（平均={avg_score:.3f}）")
    axes[idx].set_xlabel("シルエットスコア")
    axes[idx].set_ylabel("クラスタ")
    axes[idx].set_xlim(-0.2, 1)
    axes[idx].legend()

plt.tight_layout()
plt.show()