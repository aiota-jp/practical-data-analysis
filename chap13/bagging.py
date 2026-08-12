from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 乳がんデータの準備
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# バギング（決定木ベース）
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=50, max_samples=0.8, random_state=42, n_jobs=-1)
bagging.fit(X_train, y_train)

# 精度評価
print("=== Bagging ===")
print(f"学習スコア: {bagging.score(X_train, y_train):.4f}")
print(f"テストスコア: {bagging.score(X_test, y_test):.4f}")