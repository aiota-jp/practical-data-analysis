# 多重検定の問題の例
# 20回検定を行うと、全て帰無仮説が正しくても1回は「有意」と出る確率
alpha = 0.05
n_tests = 20
prob_at_least_one_false_positive = 1 - (1 - alpha) ** n_tests
print(f"{n_tests}回検定を行った場合:")
print(f"少なくとも1回偽陽性が出る確率: {prob_at_least_one_false_positive:.2%}")
# → 約64%！

# ボンフェローニ補正
alpha_corrected = alpha / n_tests
print(f"\nボンフェローニ補正後の有意水準: {alpha_corrected:.4f}")
print("→ 各検定のp値がこの値を下回った場合のみ有意と判断")