from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

# 乳がんデータセット（2クラス分類）
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target  # 0: 悪性, 1: 良性

print("=== 乳がんデータセット ===")
print(f"データ数: {X_cancer.shape[0]}, 特徴量数: {X_cancer.shape[1]}")
print(f"クラス: {cancer.target_names}")
print(f"クラスごとの件数: 悪性={(y_cancer == 0).sum()}, 良性={(y_cancer == 1).sum()}")

# 学習データとテストデータに分割
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# ロジスティック回帰モデルの構築
lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_c)

# 予測
y_pred = lr.predict(X_test_scaled)

# 精度評価
print("\n=== ロジスティック回帰の結果 ===")
print(f"正解率（Accuracy）: {accuracy_score(y_test_c, y_pred):.4f}")
print("\n=== 分類レポート ===")
print(classification_report(y_test_c, y_pred, target_names=cancer.target_names))