import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 手書き数字データの準備
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# PCAで64次元から2次元に削減
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# 結果表示
cum_ratio = pca.explained_variance_ratio_.sum()

print("=== 手書き数字データのPCA ===")
print(f"元の次元数: {X_digits.shape[1]}")
print(f"削減後の次元数: {X_2d.shape[1]}")
print(f"第1主成分の寄与率: {pca.explained_variance_ratio_[0]:.4f}")
print(f"第2主成分の寄与率: {pca.explained_variance_ratio_[1]:.4f}")
print(f"累積寄与率: {cum_ratio:.4f}")

# 可視化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_digits, cmap="tab10", s=10, alpha=0.6)
plt.colorbar(scatter, label="数字")
plt.xlabel("第1主成分")
plt.ylabel("第2主成分")
plt.title("手書き数字データのPCA可視化（64次元 → 2次元）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()