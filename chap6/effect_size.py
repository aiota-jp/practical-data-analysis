import numpy as np

def cohens_d(group1, group2):
    """Cohen's d（効果量）を計算"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    
    # プールされた標準偏差
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    return (group1.mean() - group2.mean()) / pooled_std

# 先ほどの理系・文系のデータで効果量を計算
science_scores = np.array([78, 85, 92, 88, 76, 90, 84, 95, 82, 87,
                           91, 79, 86, 93, 80, 88, 84, 90, 77, 85])
liberal_arts_scores = np.array([65, 72, 58, 70, 68, 75, 62, 71, 66, 73,
                                60, 69, 74, 63, 67, 70, 64, 72, 61, 68])

d = cohens_d(science_scores, liberal_arts_scores)
print(f"Cohen's d: {d:.4f}")

if abs(d) < 0.2:
    print("→ 効果量: ほとんどなし")
elif abs(d) < 0.5:
    print("→ 効果量: 小")
elif abs(d) < 0.8:
    print("→ 効果量: 中")
else:
    print("→ 効果量: 大")