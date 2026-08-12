import pandas as pd

df = pd.DataFrame({
    "name": ["田中", "鈴木", "田中", "佐藤", "鈴木"],
    "age": [25, 30, 25, 28, 30],
    "city": ["東京", "大阪", "東京", "名古屋", "大阪"]
})

print(df)

# 重複行の確認
print(df.duplicated())           # 重複している行をTrue/Falseで表示
print(df.duplicated().sum())     # 重複行の数

# 特定の列で重複を確認
print(df.duplicated(subset=["name"]))