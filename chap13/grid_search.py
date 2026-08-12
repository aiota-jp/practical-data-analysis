from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# データ準備
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target)

# パイプライン
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])

# パラメータ候補
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
    "svc__kernel": ["rbf", "linear"]
}

# グリッドサーチ
grid_search = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 結果
print("=== グリッドサーチ結果 ===")
print(f"最適パラメータ: {grid_search.best_params_}")
print(f"最高CVスコア: {grid_search.best_score_:.4f}")
print(f"テストスコア: {grid_search.score(X_test, y_test):.4f}")

# デフォルトパラメータとの比較
svc_default = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])
svc_default.fit(X_train, y_train)

print(f"\nデフォルトSVMのテストスコア: {svc_default.score(X_test, y_test):.4f}")
print(f"グリッドサーチ後のテストスコア: {grid_search.score(X_test, y_test):.4f}")