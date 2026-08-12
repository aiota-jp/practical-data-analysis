import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib

# データ準備
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# スケーリング（ニューラルネットワークでは必須）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLPモデルの構築
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 30),   # 中間層: 50ニューロン + 30ニューロン
    activation='relu',              # 活性化関数: ReLU
    solver='adam',                   # 最適化: Adam
    alpha=0.001,                    # 正則化パラメータ
    max_iter=1000,                  # 最大エポック数
    tol=0.00001,                    # 収束判定
    random_state=42,
    verbose=False
)

# 学習
mlp.fit(X_train_scaled, y_train)

# 予測と評価
y_pred = mlp.predict(X_test_scaled)

print(f"=== ニューラルネットワーク（MLP）による分類結果 ===")
print(f"正解率: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))