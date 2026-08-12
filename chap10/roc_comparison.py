import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

# モデルの定義
models = {
    "ロジスティック回帰": LogisticRegression(max_iter=1000, random_state=42),
    "SVM（RBF）": SVC(kernel="rbf", probability=True, random_state=42),
    "ランダムフォレスト": RandomForestClassifier(n_estimators=100, random_state=42)
}

# ROC曲線の比較
plt.figure(figsize=(8, 6))

for name, model in models.items():
    if name in ["ロジスティック回帰", "SVM（RBF）"]:
        model.fit(X_train_scaled, y_train_c)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_c, y_train_c)
        y_prob = model.predict_proba(X_test_c)[:, 1]

    fpr, tpr, _ = roc_curve(y_test_c, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name}（AUC={roc_auc:.4f}）")

plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ランダム分類器")
plt.xlabel("偽陽性率（FPR）")
plt.ylabel("真陽性率（TPR）")
plt.title("ROC曲線による分類モデルの比較")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()