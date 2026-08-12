from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

# 乳がんデータの準備
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# 精度評価
print("=== Boosting ===")
print(f"AdaBoost 学習スコア: {ada.score(X_train, y_train):.4f}")
print(f"AdaBoost テストスコア: {ada.score(X_test, y_test):.4f}")
print(f"GradientBoosting 学習スコア: {gb.score(X_train, y_train):.4f}")
print(f"GradientBoosting テストスコア: {gb.score(X_test, y_test):.4f}")