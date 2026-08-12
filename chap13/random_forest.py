import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 乳がんデータの準備
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ランダムフォレスト
rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 予測
y_pred_rf = rf.predict(X_test)

# 精度評価
print("=== ランダムフォレスト ===")
print(f"学習スコア: {rf.score(X_train, y_train):.4f}")
print(f"テストスコア: {rf.score(X_test, y_test):.4f}")

# 分類レポート
print("\n=== 分類レポート ===")
print(classification_report(y_test, y_pred_rf, target_names=cancer.target_names))

# 特徴量の重要度
importances = pd.DataFrame({"特徴量": cancer.feature_names, "重要度": rf.feature_importances_}).sort_values("重要度", ascending=False)

print("\n=== 特徴量の重要度（上位5件） ===")
print(importances.head().to_string(index=False))