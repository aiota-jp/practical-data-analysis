import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# 乳がんデータセット
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target  # 0: 悪性, 1: 良性

# 学習データとテストデータに分割
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# ロジスティック回帰
lr = LogisticRegression(C=1.0, l1_ratio=0, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_c)

# 良性（クラス1）の予測確率
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# ROC曲線とAUCの計算
fpr, tpr, thresholds = roc_curve(y_test_c, y_proba_lr)
roc_auc = auc(fpr, tpr)

# ROC曲線の描画
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC曲線（AUC = {roc_auc:.4f}）")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="ランダム分類器")
plt.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
plt.xlabel("偽陽性率（FPR）")
plt.ylabel("真陽性率（TPR）")
plt.title("ROC曲線（ロジスティック回帰）")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"AUC: {roc_auc:.4f}")