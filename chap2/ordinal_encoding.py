import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# サンプルデータの作成
df = pd.DataFrame({
    "education": ["高校", "大学", "大学院", "中学", "大学"]
})

print("===== 変換前 =====")
print(df)


# ========================================
# カテゴリの順序を定義
# ========================================

order = [
    ["中学", "高校", "大学", "大学院"]
]

encoder = OrdinalEncoder(categories=order)


# ========================================
# 順序ラベルエンコード
# ========================================

df["education_encoded"] = encoder.fit_transform(
    df[["education"]]
).astype(int)

print("\n===== 順序ラベルエンコード後 =====")
print(df)