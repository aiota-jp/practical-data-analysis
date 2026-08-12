import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 乳がんデータセット
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# 学習データとテストデータに分割
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# ロジスティック回帰
lr = LogisticRegression(C=1.0, l1_ratio=0, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_c)
y_pred_lr = lr.predict(X_test_scaled)

# 混同行列
cm = confusion_matrix(y_test_c, y_pred_lr)

# ヒートマップで可視化
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.xlabel("予測ラベル")
plt.ylabel("正解ラベル")
plt.title("混同行列（ロジスティック回帰）")
plt.tight_layout()
plt.show()

# 分類レポート
print("=== 分類レポート ===")
print(classification_report(y_test_c, y_pred_lr, target_names=cancer.target_names))