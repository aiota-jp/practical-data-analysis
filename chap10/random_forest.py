from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 乳がんデータセット
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0: 悪性, 1: 良性

# 学習データとテストデータに分割
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ランダムフォレストモデル
rf = RandomForestClassifier(
    n_estimators=100,        # 決定木の本数
    max_depth=None,          # 深さ制限なし
    min_samples_leaf=2,      # 葉の最小サンプル数
    max_features='sqrt',     # 各木で使う特徴量数（√特徴量数）
    random_state=42,
    n_jobs=-1                # 全CPUコアを使用
)
rf.fit(X_train_c, y_train_c)

# 予測
y_pred_rf = rf.predict(X_test_c)

# 精度評価
print("=== ランダムフォレストの結果（乳がんデータ） ===")
print(f"正解率: {accuracy_score(y_test_c, y_pred_rf):.4f}")
print("\n=== 分類レポート ===")
print(classification_report(y_test_c, y_pred_rf, target_names=cancer.target_names))