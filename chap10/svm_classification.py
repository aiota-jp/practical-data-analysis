from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Irisデータセット
iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータに分割
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_i_scaled = scaler.fit_transform(X_train_i)
X_test_i_scaled = scaler.transform(X_test_i)

# 線形SVM
svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
svm_linear.fit(X_train_i_scaled, y_train_i)
y_pred_linear = svm_linear.predict(X_test_i_scaled)

# RBFカーネルSVM
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_rbf.fit(X_train_i_scaled, y_train_i)
y_pred_rbf = svm_rbf.predict(X_test_i_scaled)

# 精度評価
print("=== SVM（Irisデータ: 多クラス分類） ===")
print(f"線形SVM 正解率: {accuracy_score(y_test_i, y_pred_linear):.4f}")
print(f"RBF SVM 正解率: {accuracy_score(y_test_i, y_pred_rbf):.4f}")
print("\n=== RBF SVMの分類レポート ===")
print(classification_report(y_test_i, y_pred_rbf, target_names=iris.target_names))