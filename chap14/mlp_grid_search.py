from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Irisデータの準備
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# パイプライン
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=1000, random_state=42))
])

# パラメータ候補
param_grid = {
    "mlp__hidden_layer_sizes": [(50,), (100,), (50, 30), (100, 50)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [0.0001, 0.001, 0.01]
}

# グリッドサーチ
grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# 結果
print("=== グリッドサーチ結果 ===")
print(f"最適パラメータ: {grid_search.best_params_}")
print(f"最高スコア（CV）: {grid_search.best_score_:.4f}")
print(f"テストスコア: {grid_search.score(X_test, y_test):.4f}")