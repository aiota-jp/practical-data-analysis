import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Irisデータセット
iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータに分割
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 決定木モデル
dt = DecisionTreeClassifier(
    criterion='gini',       # 不純度指標
    max_depth=4,            # 木の最大深さ（過学習防止）
    min_samples_leaf=5,     # 葉の最小サンプル数
    random_state=42
)
dt.fit(X_train_i, y_train_i)

# 予測
y_pred_dt = dt.predict(X_test_i)

# 精度評価
print("=== 決定木の結果 ===")
print(f"正解率: {accuracy_score(y_test_i, y_pred_dt):.4f}")

# 特徴量の重要度
importances = pd.DataFrame({"特徴量": iris.feature_names, "重要度": dt.feature_importances_}).sort_values("重要度", ascending=False)

print("\n=== 特徴量の重要度 ===")
print(importances.to_string(index=False))