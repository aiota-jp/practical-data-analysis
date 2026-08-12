import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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

# 全パラメータ組み合わせの結果
results_df = pd.DataFrame(grid_search.cv_results_)
top10 = results_df.nsmallest(10, "rank_test_score")[["params", "mean_test_score", "std_test_score", "rank_test_score"]]

print("=== 上位10パラメータ ===")
print(top10.to_string(index=False))