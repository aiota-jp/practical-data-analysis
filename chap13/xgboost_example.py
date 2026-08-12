from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 乳がんデータの準備
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,   # 決定木の本数
    max_depth=3,        # 木の最大深さ
    learning_rate=0.1,  # 学習率
    random_state=42
)
xgb.fit(X_train, y_train)

# 精度評価
print("=== XGBoost ===")
print(f"学習スコア: {xgb.score(X_train, y_train):.4f}")
print(f"テストスコア: {xgb.score(X_test, y_test):.4f}")