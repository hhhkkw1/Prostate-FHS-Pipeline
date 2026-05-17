# ==============================================================================
# 0. 环境准备
# ==============================================================================
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# ==============================================================================
# 1. 读取数据
# ==============================================================================
FILE_PATH = r"I:\RP_multib\重新修改\副本ALL911改10.xlsx"

print(f"📂 正在读取数据: {FILE_PATH}")

df = pd.read_excel(FILE_PATH)

# ==============================================================================
# 2. 检查必要列
# ==============================================================================
required_cols = [
    'Dataset_Type',
    'AP_Status',
    'age',
    'PSA',
    'T_score',
    'GG_BP',
    'PIRADS',
    'pct_pos',
    'Habitat_score',
    'CAPRA_score'
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ 缺少列: {col}")

# ==============================================================================
# 3. 缺失值处理
# ==============================================================================
print("🧹 正在处理缺失值...")

continuous_cols = [
    'age',
    'PSA',
    'pct_pos',
    'Habitat_score',
    'CAPRA_score',
    'T_score',
    'GG_BP',
    'PIRADS'
]

# 强制转数值
for col in continuous_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 中位数填充
for col in continuous_cols:
    med = df[col].median()
    df[col] = df[col].fillna(med)

# 结局转整数
df['AP_Status'] = pd.to_numeric(df['AP_Status'], errors='coerce')
df['AP_Status'] = df['AP_Status'].fillna(0).astype(int)

print("✅ 缺失值处理完成")

# ==============================================================================
# 4. 连续变量形式建模
# ==============================================================================
# 注意：
# 这里不再进行 one-hot 编码
# GG_BP / PIRADS / T_score 全部作为连续变量
# ==============================================================================

# ------------------------------------------------------------------------------
# Clinical
# ------------------------------------------------------------------------------
features_clin = [
    'age',
    'PSA',
    'pct_pos',
    'T_score',
    'GG_BP',
    'PIRADS'
]

# ------------------------------------------------------------------------------
# Habitat
# ------------------------------------------------------------------------------
features_habitat = [
    'Habitat_score'
]

# ------------------------------------------------------------------------------
# Combined
# ------------------------------------------------------------------------------
features_comb = features_clin + features_habitat

# ------------------------------------------------------------------------------
# CAPRA
# ------------------------------------------------------------------------------
features_capra = [
    'CAPRA_score'
]

# ==============================================================================
# 5. 划分训练集
# ==============================================================================
train_df = df[df['Dataset_Type'] == 'Training'].copy()

y_train = train_df['AP_Status']

# ==============================================================================
# 6. LogisticRegression建模
# ==============================================================================
print("🏗️ 正在训练模型...")

lr_clin = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

lr_hab = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

lr_comb = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

lr_capra = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

# ------------------------------------------------------------------------------
# 拟合
# ------------------------------------------------------------------------------
lr_clin.fit(train_df[features_clin], y_train)

lr_hab.fit(train_df[features_habitat], y_train)

lr_comb.fit(train_df[features_comb], y_train)

lr_capra.fit(train_df[features_capra], y_train)

print("✅ 模型训练完成")

# ==============================================================================
# 7. 预测概率
# ==============================================================================
print("🔮 正在生成预测概率...")

df['pred_prob_Clinical'] = lr_clin.predict_proba(
    df[features_clin]
)[:, 1]

df['pred_prob_Habitat'] = lr_hab.predict_proba(
    df[features_habitat]
)[:, 1]

df['pred_prob_Combined'] = lr_comb.predict_proba(
    df[features_comb]
)[:, 1]

df['pred_prob_CAPRA'] = lr_capra.predict_proba(
    df[features_capra]
)[:, 1]

# ==============================================================================
# 8. 保存预测结果
# ==============================================================================
OUTPUT_PATH = FILE_PATH.replace(
    ".xlsx",
    "_Predictions_Continuous.xlsx"
)

print(f"💾 正在保存预测结果: {OUTPUT_PATH}")

df.to_excel(OUTPUT_PATH, index=False)

print("✅ 预测结果保存完成")

# ==============================================================================
# 9. 提取 Logistic 回归参数
# ==============================================================================
def extract_logistic_params(X, y, model_name):

    print(f"\n📊 正在计算 {model_name} Logistic参数")

    # --------------------------------------------------------------------------
    # 强制转float
    # --------------------------------------------------------------------------
    X = X.copy().astype(float)

    y = y.copy().astype(int)

    # --------------------------------------------------------------------------
    # 加截距
    # --------------------------------------------------------------------------
    X_sm = sm.add_constant(X)

    # --------------------------------------------------------------------------
    # Logit
    # --------------------------------------------------------------------------
    model = sm.Logit(y, X_sm)

    result = model.fit(disp=False)

    # --------------------------------------------------------------------------
    # 提取参数
    # --------------------------------------------------------------------------
    params = result.params

    conf = result.conf_int()

    pvals = result.pvalues

    summary_df = pd.DataFrame({

        'Model': model_name,

        'Variable': params.index,

        'Beta': params.values,

        'OR': np.exp(params.values),

        'CI_lower': np.exp(conf[0].values),

        'CI_upper': np.exp(conf[1].values),

        'p_value': pvals.values
    })

    # 去掉截距
    summary_df = summary_df[
        summary_df['Variable'] != 'const'
    ]

    return summary_df

# ==============================================================================
# 10. 提取所有模型参数
# ==============================================================================
params_all = []

# ------------------------------------------------------------------------------
# Clinical
# ------------------------------------------------------------------------------
params_all.append(
    extract_logistic_params(
        train_df[features_clin],
        y_train,
        "Clinical"
    )
)

# ------------------------------------------------------------------------------
# Habitat
# ------------------------------------------------------------------------------
params_all.append(
    extract_logistic_params(
        train_df[features_habitat],
        y_train,
        "Habitat"
    )
)

# ------------------------------------------------------------------------------
# Combined
# ------------------------------------------------------------------------------
params_all.append(
    extract_logistic_params(
        train_df[features_comb],
        y_train,
        "Combined"
    )
)

# ------------------------------------------------------------------------------
# CAPRA
# ------------------------------------------------------------------------------
params_all.append(
    extract_logistic_params(
        train_df[features_capra],
        y_train,
        "CAPRA"
    )
)

# ==============================================================================
# 11. 合并结果
# ==============================================================================
params_df = pd.concat(
    params_all,
    ignore_index=True
)

# ==============================================================================
# 12. 保存Logistic参数
# ==============================================================================
PARAM_OUTPUT_PATH = FILE_PATH.replace(
    ".xlsx",
    "_Logistic_Params_Continuous.xlsx"
)

print(f"💾 正在保存参数结果: {PARAM_OUTPUT_PATH}")

params_df.to_excel(
    PARAM_OUTPUT_PATH,
    index=False
)

print("✅ Logistic参数保存完成")

# ==============================================================================
# 13. 输出模型参数
# ==============================================================================
print("\n================ Logistic Parameters ================\n")

print(params_df)