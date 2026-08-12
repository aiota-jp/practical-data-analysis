import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Irisデータの準備
iris = load_iris()
X = iris.data

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# エルボー法
sse_list = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    sse_list.append(km.inertia_)

# 結果表示
print("=== クラスタ数とSSE ===")
for k, sse in zip(k_range, sse_list):
    print(f"K={k}: SSE={sse:.2f}")

# 可視化
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse_list, "o-", color="steelblue", markersize=8, linewidth=2)
plt.xlabel("クラスタ数 K")
plt.ylabel("SSE（クラスタ内誤差平方和）")
plt.title("エルボー法によるクラスタ数の決定")
plt.axvline(3, color="red", linestyle="--", alpha=0.7, label="エルボーポイント（K=3）")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(k_range)
plt.show()

print("\nK=3付近からSSEの減少が緩やかになるため、クラスタ数3が候補となります。")