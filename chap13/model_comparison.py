from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# 乳がんデータの準備
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 各モデルを5分割交差検証で比較
print("=== アンサンブル手法の比較（乳がんデータ: 5分割CV） ===")
print(f"{'モデル':<25} {'平均スコア':>10} {'標準偏差':>10}")
print("-" * 50)

for name, model in [
    ("ロジスティック回帰", LogisticRegression(max_iter=1000)),
    ("SVM（RBF）", SVC()),
    ("決定木", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("ランダムフォレスト", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42))
]:
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
    print(f"{name:<25} {scores.mean():>10.4f} {scores.std():>10.4f}")