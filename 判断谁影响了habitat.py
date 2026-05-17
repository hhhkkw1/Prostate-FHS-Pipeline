import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import statsmodels.api as sm
from scipy.stats import chi2

# =========================================================
# 1. 读取数据
# =========================================================

file_path = r"I:\RP_multib\重新修改\副本ALL911改3.xlsx"

df = pd.read_excel(file_path)

# =========================================================
# 2. 设定结局变量
# =========================================================

target = "AP_Status"

# =========================================================
# 3. 自定义模型参数
#    你只需要修改这里
# =========================================================

model_dict = {

    # -------------------------
    # Habitat单独模型
    # -------------------------
    "Habitat_only": [
        "Habitat_score"
    ],

    # -------------------------
    # Clinical单变量
    # -------------------------
    "Habitat_PSA": [
        "Habitat_score",
        "PSA"
    ],

    "Habitat_GG": [
        "Habitat_score",
        "GG_BP"
    ],

    "Habitat_PIRADS": [
        "Habitat_score",
        "PIRADS"
    ],

    "Habitat_Tstage": [
        "Habitat_score",
        "T_score"
    ],

    # -------------------------
    # 组合模型
    # -------------------------
    "Habitat_GG_PIRADS": [
        "Habitat_score",
        "GG_BP",
        "PIRADS"
    ],

    "Habitat_Full_Clinical": [
        "Habitat_score",
        "age",
        "PSA",
        "pct_pos",
        "GG_BP",
        "PIRADS",
        "T_score"
    ]
}

# =========================================================
# 4. 数据预处理
# =========================================================

# 如果GG和PIRADS是分类字符串，可转为连续数值
# 这里按需修改

# GG
if df["GG_BP"].dtype == object:
    gg_map = {
        "1":1,
        "2":2,
        "3":3,
        "4":4,
        "5":5
    }
    df["GG_BP"] = df["GG_BP"].map(gg_map)

# PIRADS
if df["PIRADS"].dtype == object:
    pirads_map = {
        "3":3,
        "4":4,
        "5":5
    }
    df["PIRADS"] = df["PIRADS"].map(pirads_map)

# 删除缺失
all_vars = list(set(
    [target] + [v for sublist in model_dict.values() for v in sublist]
))

data = df[all_vars].dropna()

# =========================================================
# 5. 结果保存
# =========================================================

results = []

# baseline OR
baseline_or = None

# =========================================================
# 6. 循环模型
# =========================================================

for model_name, variables in model_dict.items():

    print("\n================================================")
    print(f"Model: {model_name}")
    print("Variables:", variables)

    X = data[variables]
    y = data[target]

    # =====================================================
    # statsmodels Logistic
    # =====================================================

    X_sm = sm.add_constant(X)

    model = sm.Logit(y, X_sm).fit(disp=False)

    # =====================================================
    # 预测
    # =====================================================

    pred_prob = model.predict(X_sm)

    pred_binary = (pred_prob >= 0.5).astype(int)

    # =====================================================
    # 指标
    # =====================================================

    auc = roc_auc_score(y, pred_prob)

    acc = accuracy_score(y, pred_binary)

    sen = recall_score(y, pred_binary)

    spe = recall_score(y, pred_binary, pos_label=0)

    f1 = f1_score(y, pred_binary)

    aic = model.aic

    bic = model.bic

    llf = model.llf

    # =====================================================
    # Habitat OR
    # =====================================================

    if "Habitat_score" in variables:

        beta = model.params["Habitat_score"]

        habitat_or = np.exp(beta)

        pval = model.pvalues["Habitat_score"]

    else:

        habitat_or = np.nan
        pval = np.nan

    # =====================================================
    # OR attenuation
    # =====================================================

    if model_name == "Habitat_only":

        baseline_or = habitat_or

        attenuation = 0

    else:

        if baseline_or is not None and not np.isnan(habitat_or):

            attenuation = (
                (baseline_or - habitat_or)
                / baseline_or
            ) * 100

        else:

            attenuation = np.nan

    # =====================================================
    # 保存
    # =====================================================

    results.append({

        "Model": model_name,

        "Variables": ", ".join(variables),

        "AUC": auc,

        "Accuracy": acc,

        "Sensitivity": sen,

        "Specificity": spe,

        "F1": f1,

        "AIC": aic,

        "BIC": bic,

        "LogLik": llf,

        "Habitat_OR": habitat_or,

        "Habitat_p": pval,

        "OR_Attenuation_%": attenuation
    })

    # =====================================================
    # 输出回归结果
    # =====================================================

    coef_table = pd.DataFrame({

        "Variable": model.params.index,

        "Beta": model.params.values,

        "OR": np.exp(model.params.values),

        "P_value": model.pvalues.values
    })

    print("\nCoefficient Table:")
    print(coef_table)

# =========================================================
# 7. 汇总结果
# =========================================================

results_df = pd.DataFrame(results)

print("\n\n================ FINAL RESULTS ================\n")
print(results_df)

# =========================================================
# 8. 保存Excel
# =========================================================

output_path = r"I:\RP_multib\重新修改\Habitat_Incremental_Analysis.xlsx"

with pd.ExcelWriter(output_path) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Model_Comparison",
        index=False
    )

print(f"\n结果已保存至:\n{output_path}")