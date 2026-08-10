import numpy as np
import pandas as pd

# サンプルデータ
df = pd.DataFrame({
    "age": [35, 19, 23, 18, 42, 54, 26, 21, 48, 61],
    "destination": ["北海道", "沖縄", "沖縄", "沖縄", "北海道", "北海道", "沖縄", "沖縄", "北海道", "北海道"]
})

def correlation_ratio(categories, values):
    """相関比（η²）を計算する関数"""
    categories = np.array(categories)
    values = np.array(values)
    
    # 全体平均
    overall_mean = values.mean()
    
    # 全体変動
    total_variation = np.sum((values - overall_mean) ** 2)
    
    # グループ間変動
    between_variation = 0
    for cat in np.unique(categories):
        group = values[categories == cat]
        group_mean = group.mean()
        between_variation += len(group) * (group_mean - overall_mean) ** 2
    
    # 相関比
    eta_squared = between_variation / total_variation
    return eta_squared

# 相関比の計算
eta2 = correlation_ratio(df["destination"], df["age"])
print(f"相関比 η²: {eta2:.4f}")

# 判定
if eta2 < 0.1:
    print("→ ほとんど相関がない")
elif eta2 < 0.3:
    print("→ 弱い相関がある")
else:
    print("→ 強い相関がある")