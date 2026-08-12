import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score

# Irisデータセット
iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータに分割
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 決定木モデル
dt = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=5, random_state=42)
dt.fit(X_train_i, y_train_i)

# 予測
y_pred_dt = dt.predict(X_test_i)

# 精度評価
print("=== 決定木の結果 ===")
print(f"正解率: {accuracy_score(y_test_i, y_pred_dt):.4f}")

# テキスト形式で表示
print("\n=== 決定木のルール ===")
print(export_text(dt, feature_names=list(iris.feature_names)))

# グラフィカル表示
plt.figure(figsize=(16, 8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, fontsize=9)
plt.title("決定木の可視化（Irisデータセット）")
plt.tight_layout()
plt.show()