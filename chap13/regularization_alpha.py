import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# alphaの値を変えて精度を比較
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (Model, name) in enumerate([(Ridge, "Ridge"), (Lasso, "Lasso")]):
    train_scores, test_scores = [], []

    for alpha in alphas:
        model = Model(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        train_scores.append(model.score(X_train_scaled, y_train))
        test_scores.append(model.score(X_test_scaled, y_test))

    axes[idx].plot(alphas, train_scores, "o-", label="学習データ", markersize=6)
    axes[idx].plot(alphas, test_scores, "o-", label="テストデータ", markersize=6)
    axes[idx].set_xscale("log")
    axes[idx].set_xlabel("alpha（正則化の強さ）")
    axes[idx].set_ylabel("R²スコア")
    axes[idx].set_title(f"{name}回帰：alphaと精度の関係")
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()