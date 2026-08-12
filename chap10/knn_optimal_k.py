import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Irisデータセット
iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータに分割
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_i_scaled = scaler.fit_transform(X_train_i)

# Kを1～30まで変えて交差検証
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train_i_scaled, y_train_i, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

# 最適なK
best_k = list(k_range)[np.argmax(cv_scores)]

# 可視化
plt.figure(figsize=(10, 5))
plt.plot(k_range, cv_scores, "o-", color="steelblue", markersize=5)
plt.axvline(best_k, color="red", linestyle="--", label=f"最適K={best_k}")
plt.xlabel("K（近傍数）")
plt.ylabel("交差検証スコア（正解率）")
plt.title("Kの値と正解率の関係")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"最適なK: {best_k}（正解率: {max(cv_scores):.4f}）")