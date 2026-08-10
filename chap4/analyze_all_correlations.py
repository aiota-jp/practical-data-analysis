import numpy as np
import pandas as pd
from scipy import stats


# ========================================
# 相関比 η² を計算する関数
# ========================================
def correlation_ratio(categories, values):
    """カテゴリ変数と数値変数の相関比 η² を計算する"""

    overall_mean = np.mean(values)

    # 全体平方和
    total_ss = np.sum((values - overall_mean) ** 2)

    # グループ間平方和
    between_ss = 0

    for category in categories.unique():
        group = values[categories == category]

        between_ss += len(group) * (
            np.mean(group) - overall_mean
        ) ** 2

    # 全データが同じ値の場合のゼロ除算対策
    if total_ss == 0:
        return 0

    return between_ss / total_ss


# ========================================
# 全変数の関連を一括分析する関数
# ========================================
def analyze_all_correlations(df, numeric_cols, categorical_cols):
    """全変数の関連を一括で分析する関数"""

    results = []

    # ====================================
    # 量的 × 量的
    # ピアソンの相関係数
    # ====================================
    for i, col1 in enumerate(numeric_cols):

        for col2 in numeric_cols[i + 1:]:

            r, p = stats.pearsonr(
                df[col1],
                df[col2]
            )

            results.append({
                "変数1": col1,
                "変数2": col2,
                "組み合わせ": "量的×量的",
                "指標": "相関係数(R)",
                "値": r,
                "p値": p
            })


    # ====================================
    # 量的 × 質的
    # 相関比 η²
    # ====================================
    for num_col in numeric_cols:

        for cat_col in categorical_cols:

            eta2 = correlation_ratio(
                df[cat_col],
                df[num_col]
            )

            results.append({
                "変数1": num_col,
                "変数2": cat_col,
                "組み合わせ": "量的×質的",
                "指標": "相関比(η²)",
                "値": eta2,
                "p値": np.nan
            })


    # ====================================
    # 質的 × 質的
    # クラメールのV
    # ====================================
    for i, col1 in enumerate(categorical_cols):

        for col2 in categorical_cols[i + 1:]:

            # クロス集計
            cross = pd.crosstab(
                df[col1],
                df[col2]
            )

            # カイ二乗検定
            chi2, p, dof, expected = stats.chi2_contingency(
                cross
            )

            # データ件数
            n_total = cross.values.sum()

            # 行数・列数のうち小さい方 - 1
            k = min(cross.shape) - 1

            # クラメールのV
            if k > 0:
                v = np.sqrt(
                    chi2 / (n_total * k)
                )
            else:
                v = 0

            results.append({
                "変数1": col1,
                "変数2": col2,
                "組み合わせ": "質的×質的",
                "指標": "クラメールのV",
                "値": v,
                "p値": p
            })


    return pd.DataFrame(results)


# ========================================
# サンプルデータ
# ========================================
np.random.seed(42)

df = pd.DataFrame({
    "age": [
        25, 32, 28, 45, 38,
        50, 29, 41, 35, 47
    ],

    "salary": [
        350, 420, 380, 600, 500,
        680, 390, 550, 460, 620
    ],

    "department": [
        "営業部", "営業部", "開発部", "開発部", "開発部",
        "管理部", "営業部", "管理部", "営業部", "管理部"
    ],

    "gender": [
        "男性", "女性", "男性", "男性", "女性",
        "男性", "女性", "女性", "男性", "女性"
    ]
})


# ========================================
# 分析実行
# ========================================
result_df = analyze_all_correlations(
    df,
    numeric_cols=["age", "salary"],
    categorical_cols=["department", "gender"]
)


# ========================================
# 結果表示
# ========================================
print("【全変数の関連分析】")

print(
    result_df
    .sort_values("値", ascending=False)
    .to_string(index=False)
)