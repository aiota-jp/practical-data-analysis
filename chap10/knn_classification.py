from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Irisデータセット
iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータに分割
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_i_scaled = scaler.fit_transform(X_train_i)
X_test_i_scaled = scaler.transform(X_test_i)

# K近傍法モデル
knn = KNeighborsClassifier(
    n_neighbors=5,           # K=5（近傍数）
    weights='uniform',       # 重み（uniform: 均一, distance: 距離に応じた重み）
    algorithm='auto'         # 最適なアルゴリズムを自動選択
)
knn.fit(X_train_i_scaled, y_train_i)

# 予測
y_pred_knn = knn.predict(X_test_i_scaled)

# 精度評価
print("=== K近傍法（K=5）の結果 ===")
print(f"正解率: {accuracy_score(y_test_i, y_pred_knn):.4f}")