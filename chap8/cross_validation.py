import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

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

# KFoldで分割方法を定義
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5分割交差検証
cv_scores_r2 = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
cv_scores_rmse = -cross_val_score(LinearRegression(), X, y, cv=kf, scoring="neg_root_mean_squared_error")

print("=== 5分割交差検証 ===")
print(f"R²:   平均={cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
print(f"RMSE: 平均={cv_scores_rmse.mean():.2f} ± {cv_scores_rmse.std():.2f}")