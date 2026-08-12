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

# 木の本数を変えて精度を確認
n_trees = [1, 5, 10, 20, 50, 100, 200, 500]
scores = []

for n in n_trees:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train_c, y_train_c)
    scores.append(rf_temp.score(X_test_c, y_test_c))

# 結果表示
print("=== 決定木の本数と正解率 ===")
for n, score in zip(n_trees, scores):
    print(f"{n:>3}本: {score:.4f}")

# 可視化
plt.figure(figsize=(8, 5))
plt.plot(n_trees, scores, "o-", color="steelblue", markersize=8)
plt.xlabel("決定木の本数")
plt.ylabel("正解率")
plt.title("ランダムフォレスト：決定木の本数と正解率の関係")
plt.grid(alpha=0.3)
plt.xscale("log")
plt.show()