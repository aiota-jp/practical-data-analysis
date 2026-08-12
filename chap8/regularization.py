import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge回帰
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Lasso回帰
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)

# 結果表示
print("=== 正則化モデルの比較 ===")
print(f"Ridge R²（テスト）: {ridge.score(X_test_scaled, y_test):.4f}")
print(f"Lasso R²（テスト）: {lasso.score(X_test_scaled, y_test):.4f}")
print(f"Lasso 係数: {lasso.coef_}")
print(f"0になった係数（不要と判断された変数）の数: {(lasso.coef_ == 0).sum()}")