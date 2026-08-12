from scipy import stats

# Webサイトのコンバージョン率が業界平均（3%）と異なるか検定
n_visitors = 500       # 訪問者数
n_conversions = 22     # コンバージョン数
industry_rate = 0.03   # 業界平均 3%

result = stats.binomtest(n_conversions, n_visitors, industry_rate, alternative='two-sided')

print(f"訪問者数: {n_visitors}")
print(f"コンバージョン数: {n_conversions}")
print(f"コンバージョン率: {n_conversions/n_visitors*100:.1f}%")
print(f"業界平均: {industry_rate*100:.1f}%")
print(f"p値: {result.pvalue:.6f}")

if result.pvalue < 0.05:
    print("→ 業界平均と統計的に有意な差がある")
else:
    print("→ 業界平均と有意な差があるとは言えない")