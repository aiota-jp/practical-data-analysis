from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 乳がんデータセット
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

# 各モデル
all_models = {
    "ロジスティック回帰": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "SVM（線形）": make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=42)),
    "SVM（RBF）": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
    "K近傍法（K=5）": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "決定木": DecisionTreeClassifier(max_depth=5, random_state=42),
    "ランダムフォレスト": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 5分割交差検証
print("=== モデル比較（5分割交差検証） ===")
print(f"{'モデル':<20} {'平均正解率':>10} {'標準偏差':>10}")
print("-" * 45)

for name, model in all_models.items():
    scores = cross_val_score(model, X_cancer, y_cancer, cv=5, scoring="accuracy")
    print(f"{name:<20} {scores.mean():>10.4f} {scores.std():>10.4f}")