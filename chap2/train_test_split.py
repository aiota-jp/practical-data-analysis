from sklearn.model_selection import train_test_split
import pandas as pd

# サンプルデータの作成
df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 28, 32, 38, 42, 50],
    "salary": [
        3000000, 4000000, 4500000, 5000000, 6000000,
        3500000, 4200000, 4800000, 5500000, 7000000
    ],
    "experience": [2, 5, 8, 12, 18, 4, 6, 10, 15, 22],
    "target": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
})

print("===== 元のデータ =====")
print(df)


# ========================================
# 特徴量（X）と目的変数（y）に分離
# ========================================

X = df.drop("target", axis=1)
y = df["target"]

print("\n===== 特徴量（X） =====")
print(X)

print("\n===== 目的変数（y） =====")
print(y)


# ========================================
# 学習データとテストデータに分割
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ========================================
# 分割結果
# ========================================

print("\n===== データ分割結果 =====")

print(f"全データ   : {len(df)}件")
print(f"学習データ : {X_train.shape[0]}件")
print(f"テストデータ: {X_test.shape[0]}件")


print("\n===== 学習データ（X_train） =====")
print(X_train)

print("\n===== 学習データの正解（y_train） =====")
print(y_train)


print("\n===== テストデータ（X_test） =====")
print(X_test)

print("\n===== テストデータの正解（y_test） =====")
print(y_test)