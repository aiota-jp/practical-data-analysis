import pandas as pd
from sklearn.preprocessing import LabelEncoder

# サンプルデータの作成
df = pd.DataFrame({
    "name": ["田中", "鈴木", "佐藤", "山田", "高橋"],
    "city": ["東京", "大阪", "東京", "名古屋", "大阪"]
})

print("===== 変換前 =====")
print(df)


# ========================================
# LabelEncoderによる変換
# ========================================

encoder = LabelEncoder()

df["city_encoded"] = encoder.fit_transform(df["city"])

print("\n===== LabelEncoderによる変換後 =====")
print(df)


# ========================================
# カテゴリと数値の対応を表示
# ========================================

print("\n===== カテゴリと数値の対応 =====")

for number, category in enumerate(encoder.classes_):
    print(f"{category} = {number}")