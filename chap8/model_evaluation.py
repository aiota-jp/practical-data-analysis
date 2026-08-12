import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import japanize_matplotlib

# 住宅データの再利用
np.random.seed(42)
n = 200
area = np.random.uniform(40, 120, n)
distance = np.random.uniform(1, 30, n)
age = np.random.uniform(0, 40, n)
rooms = np.random.randint(1, 5, n)
price = 30*area - 50*distance - 20*age + 200*rooms + 1000 + np.random.normal(0, 300, n)

X = np.column_stack([area, distance, age, rooms])
y = price

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの構築
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 評価指標の計算
print("=== モデル評価指標 ===")
print(f"\n{'指標':<15} {'学習データ':>12} {'テストデータ':>12}")
print("-" * 45)
print(f"{'R²':<15} {r2_score(y_train, y_pred_train):>12.4f} {r2_score(y_test, y_pred_test):>12.4f}")
print(f"{'RMSE':<15} {np.sqrt(mean_squared_error(y_train, y_pred_train)):>12.2f} "
      f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):>12.2f}")
print(f"{'MAE':<15} {mean_absolute_error(y_train, y_pred_train):>12.2f} "
      f"{mean_absolute_error(y_test, y_pred_test):>12.2f}")

# 過学習の判断
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
if r2_train - r2_test > 0.1:
    print("\n⚠️ 学習データとテストデータのR²に大きな差があります（過学習の可能性）")
else:
    print("\n✓ 過学習の兆候は見られません")