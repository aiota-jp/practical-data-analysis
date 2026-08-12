import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# カリフォルニア住宅価格データ
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# データ分割とスケーリング
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 通常の線形回帰
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Lasso回帰（L1正則化）
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)

# 比較
print("=== 線形回帰 vs Lasso回帰 ===")
print(f"{'モデル':<15} {'学習R²':>8} {'テストR²':>8} {'RMSE':>8}")
print("-" * 45)

for name, model in [("線形回帰", lr), ("Lasso(α=0.01)", lasso)]:
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))
    print(f"{name:<15} {train_r2:>8.4f} {test_r2:>8.4f} {rmse:>8.4f}")

# Lassoの係数
print("\n=== Lasso回帰の係数 ===")
for feature, coef in zip(housing.feature_names, lasso.coef_):
    print(f"{feature:<10}: {coef:.4f}")

print(f"\n0になった係数の数: {(lasso.coef_ == 0).sum()}")