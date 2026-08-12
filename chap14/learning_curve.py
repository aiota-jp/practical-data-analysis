import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Irisデータの準備
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ニューラルネットワークモデル
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# 予測
y_pred = mlp.predict(X_test_scaled)

# 精度評価
print("=== MLPの分類結果 ===")
print(f"正解率: {accuracy_score(y_test, y_pred):.4f}")
print("\n=== 分類レポート ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 損失関数の推移を可視化
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_, color="steelblue", linewidth=2)
plt.xlabel("反復回数")
plt.ylabel("損失（Loss）")
plt.title("学習曲線（損失関数の推移）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n最終損失: {mlp.loss_:.6f}")
print(f"学習反復回数: {mlp.n_iter_}")