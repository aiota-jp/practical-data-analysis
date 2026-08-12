from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

# スタッキングで使用するモデル
estimators = [
    ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("svc", SVC(probability=True, random_state=42)),
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42))
]

# スタッキング
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking.fit(X_train_s, y_train)

# 精度評価
print("=== Stacking ===")
print(f"学習スコア: {stacking.score(X_train_s, y_train):.4f}")
print(f"テストスコア: {stacking.score(X_test_s, y_test):.4f}")