import pandas as pd

# サンプルデータの作成
df = pd.DataFrame({
    "name": ["田中", "鈴木", "佐藤", "山田"],
    "blood_type": ["A", "B", "O", "AB"],
    "city": ["東京", "大阪", "東京", "名古屋"]
})

print("===== 元のデータ =====")
print(df)


# ========================================
# 1. One-Hotエンコード
# ========================================

df_encoded = pd.get_dummies(
    df,
    columns=["blood_type", "city"],
    dtype=int
)

print("\n===== One-Hotエンコード後 =====")
print(df_encoded)


# ========================================
# 2. drop_first=Trueを指定
# ========================================

df_drop_first = pd.get_dummies(
    df,
    columns=["blood_type"],
    drop_first=True,
    dtype=int
)

print("\n===== drop_first=Trueを指定した結果 =====")
print(df_drop_first)