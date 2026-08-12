import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 乳がんデータセット
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 学習データとテストデータに分割
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ランダムフォレストモデル
rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
rf.fit(X_train_c, y_train_c)

# 特徴量の重要度
importances_rf = pd.DataFrame({"特徴量": cancer.feature_names, "重要度": rf.feature_importances_}).sort_values("重要度", ascending=False)

print("=== 特徴量の重要度（上位15件） ===")
print(importances_rf.head(15).to_string(index=False))

# 上位15件を可視化
top15 = importances_rf.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top15["特徴量"][::-1], top15["重要度"][::-1], color="steelblue", alpha=0.8)
plt.xlabel("重要度")
plt.title("ランダムフォレスト：特徴量の重要度（上位15件）")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()