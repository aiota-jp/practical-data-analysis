from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import numpy as np

# 手書き数字データの読み込み
digits = load_digits()
print(f"=== 手書き数字データセット ===")
print(f"データ数: {digits.data.shape[0]}")
print(f"特徴量数: {digits.data.shape[1]}（8×8ピクセル）")
print(f"クラス数: {len(digits.target_names)}（0〜9）")

# データのサンプル表示
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"ラベル: {digits.target[i]}")
    ax.axis('off')
plt.suptitle("手書き数字データのサンプル", fontsize=13)
plt.tight_layout()
plt.show()

# データ分割とスケーリング
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLPモデルの構築と学習
mlp_digits = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp_digits.fit(X_train_scaled, y_train)

# 評価
y_pred = mlp_digits.predict(X_test_scaled)
print(f"\n正解率: {accuracy_score(y_test, y_pred):.4f}")

# 混同行列
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel("予測ラベル")
plt.ylabel("正解ラベル")
plt.title(f"混同行列（正解率: {accuracy_score(y_test, y_pred):.4f}）")
plt.tight_layout()
plt.show()