from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Irisデータの準備
iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 学習の終了条件を設定
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=10000,     # 最大反復回数
    tol=0.00001,        # 改善量の判定基準
    random_state=42
)

# 学習
mlp.fit(X_train_scaled, y_train)

# 結果
print("=== MLPの学習結果 ===")
print(f"実際の反復回数: {mlp.n_iter_}")
print(f"最終損失: {mlp.loss_:.6f}")
print(f"テストスコア: {mlp.score(X_test_scaled, y_test):.4f}")

# 学習終了の確認
if mlp.n_iter_ < mlp.max_iter:
    print("→ 収束条件を満たして学習終了")
else:
    print("→ max_iterに達して学習終了（収束していない可能性あり）")