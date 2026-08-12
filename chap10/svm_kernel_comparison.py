import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Irisデータセット
iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータに分割
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_i_scaled = scaler.fit_transform(X_train_i)

# 2特徴量のみ使用（可視化のため）
X_2d = X_train_i_scaled[:, 2:4]  # petal length, petal width
y_2d = y_train_i

# カーネルごとの決定境界を可視化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
kernels = ["linear", "poly", "rbf"]

for ax, kernel in zip(axes, kernels):
    svm = SVC(kernel=kernel, C=1.0, gamma="scale")
    svm.fit(X_2d, y_2d)

    # 決定境界の描画
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="Set2")
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap="Set2", edgecolors="black", s=30)
    ax.set_title(f"カーネル: {kernel}\n学習データ正解率: {svm.score(X_2d, y_2d):.3f}")
    ax.set_xlabel("petal length（標準化）")
    ax.set_ylabel("petal width（標準化）")

plt.tight_layout()
plt.show()