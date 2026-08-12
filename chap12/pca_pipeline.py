from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Irisデータの準備
iris = load_iris()
X = iris.data
y = iris.target

# PCA + ロジスティック回帰のパイプライン
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 5分割交差検証
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

print("=== PCA（2次元）+ ロジスティック回帰 ===")
print(f"正解率: {scores.mean():.4f} ± {scores.std():.4f}")

# 主成分数を変えて比較
print("\n=== 主成分数と正解率の関係 ===")

for n in [1, 2, 3, 4]:
    pipe.set_params(pca__n_components=n)
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"{n}次元: {scores.mean():.4f} ± {scores.std():.4f}")