import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# カリフォルニア住宅価格データ
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PLS回帰
pls = PLSRegression(n_components=5)
pls.fit(X_train_scaled, y_train)

# 精度評価
pls_train_r2 = pls.score(X_train_scaled, y_train)
pls_test_r2 = pls.score(X_test_scaled, y_test)

print("=== PLS回帰（n_components=5） ===")
print(f"学習R²: {pls_train_r2:.4f}")
print(f"テストR²: {pls_test_r2:.4f}")

# コンポーネント数を変えて比較
print("\n=== コンポーネント数とテストR² ===")

for n in range(1, 9):
    pls_temp = PLSRegression(n_components=n)
    pls_temp.fit(X_train_scaled, y_train)
    test_r2 = pls_temp.score(X_test_scaled, y_test)
    print(f"n_components={n}: テストR²={test_r2:.4f}")