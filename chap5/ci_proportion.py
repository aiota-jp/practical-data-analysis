import numpy as np
from scipy import stats

# 実践例: 選挙の出口調査
# 100人に調査した結果、65人がA候補を支持
n = 100          # 標本サイズ
k = 65           # 支持者数
p_hat = k / n    # 標本比率

# 正規近似による95%信頼区間
confidence = 0.95
z = stats.norm.ppf((1 + confidence) / 2)
se = np.sqrt(p_hat * (1 - p_hat) / n)

lower = p_hat - z * se
upper = p_hat + z * se

print(f"=== 母比率の区間推定（正規近似） ===")
print(f"標本サイズ n: {n}")
print(f"支持者数: {k}")
print(f"標本比率 p̂: {p_hat:.4f} ({p_hat*100:.1f}%)")
print(f"標準誤差 SE: {se:.4f}")
print(f"95%信頼区間: [{lower:.4f}, {upper:.4f}]")
print(f"           = [{lower*100:.1f}%, {upper*100:.1f}%]")
print()

# 判定: 50%を超えるか
if lower > 0.5:
    print("→ 95%信頼区間の下限が50%を超えているため、")
    print("  当選の可能性が高いと判断できます。")
else:
    print("→ 95%信頼区間の下限が50%以下のため、")
    print("  当選するかどうかは断言できません。")