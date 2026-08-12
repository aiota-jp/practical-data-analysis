import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# サンプルデータ（住宅データ）
np.random.seed(42)
n = 200
area = np.random.uniform(40, 120, n)
distance = np.random.uniform(1, 30, n)
age = np.random.uniform(0, 40, n)
rooms = np.random.randint(1, 5, n)
price = 30 * area - 50 * distance - 20 * age + 200 * rooms + 1000 + np.random.normal(0, 300, n)

# 説明変数と目的変数
X = np.column_stack([area, distance, age, rooms])
y = price

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの学習
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# R²の計算
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("=== モデル評価 ===")
print(f"学習データ R²: {r2_train:.4f}")
print(f"テストデータ R²: {r2_test:.4f}")

# 実測値と予測値の可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 学習データ
axes[0].scatter(y_train, y_pred_train, alpha=0.6, edgecolors="black", s=40)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", linewidth=2, label="完全な予測")
axes[0].set_xlabel("実測値（万円）")
axes[0].set_ylabel("予測値（万円）")
axes[0].set_title(f"学習データ（R²={r2_train:.4f}）")
axes[0].legend()
axes[0].grid(alpha=0.3)

# テストデータ
axes[1].scatter(y_test, y_pred_test, alpha=0.6, edgecolors="black", s=40, color="coral")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="完全な予測")
axes[1].set_xlabel("実測値（万円）")
axes[1].set_ylabel("予測値（万円）")
axes[1].set_title(f"テストデータ（R²={r2_test:.4f}）")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()