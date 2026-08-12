import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 乳がんデータセット
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target  # 0: 悪性, 1: 良性

# 学習データとテストデータに分割
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# ロジスティック回帰モデルの構築
lr = LogisticRegression(C=1.0, l1_ratio=0, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_c)

# 予測
y_pred = lr.predict(X_test_scaled)

# 精度評価
print("=== ロジスティック回帰の結果 ===")
print(f"正解率（Accuracy）: {accuracy_score(y_test_c, y_pred):.4f}")
print("\n=== 分類レポート ===")
print(classification_report(y_test_c, y_pred, target_names=cancer.target_names))

# 回帰係数（影響度）の確認
coef_df = pd.DataFrame({"特徴量": cancer.feature_names, "係数": lr.coef_[0]})
coef_df["係数の絶対値"] = coef_df["係数"].abs()
coef_df = coef_df.sort_values("係数の絶対値", ascending=False)

print("\n=== 影響度の大きい特徴量（上位10件） ===")
print(coef_df.head(10)[["特徴量", "係数"]].to_string(index=False))

# 可視化
top10 = coef_df.head(10)
colors = ["red" if c < 0 else "blue" for c in top10["係数"]]
plt.figure(figsize=(10, 6))
plt.barh(top10["特徴量"], top10["係数"], color=colors, alpha=0.7)
plt.xlabel("係数（正: 良性に寄与、負: 悪性に寄与）")
plt.title("ロジスティック回帰の係数（影響度上位10件）")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()