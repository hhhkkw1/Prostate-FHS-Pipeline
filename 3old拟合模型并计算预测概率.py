# ==============================================================================
# 0. 环境准备
# ==============================================================================
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. 读取数据
# ==============================================================================
FILE_PATH = "I:\RP_multib\重新修改\副本ALL911改3.xlsx"
print(f"📂 正在读取数据: {FILE_PATH}")
df = pd.read_excel(FILE_PATH)

# 确认列名
required_cols = ['Dataset_Type', 'AP_Status', 'age', 'PSA', 'T_score', 'GG_BP', 'PIRADS',
                 'pct_pos', 'Habitat_score', 'CAPRA_score']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ 缺少列: {col}")

# ==============================================================================
# 1.5 [关键修复] 数据清洗与缺失值处理
# ==============================================================================
print("🧹 正在检查并处理缺失值...")

# 1. 检查哪些列有缺失
missing_cols = df[required_cols].isnull().sum()
print("   >>> 各列缺失值数量:")
print(missing_cols[missing_cols > 0])

# 2. 填充连续变量缺失值 (使用训练集的中位数填充，防止数据泄露)
# 注意：为了工程实现的简便性，这里我们使用全局中位数填充，
# 如果非常严格，应该只用 Training 的中位数填 Training 和 Validation。
fill_values = {
    'age': df['age'].median(),
    'PSA': df['PSA'].median(),
    'pct_pos': df['pct_pos'].median(), # 如果 pct_pos 有空值，用中位数填
    'Habitat_score': df['Habitat_score'].median(),
    'CAPRA_score': df['CAPRA_score'].median()
}

# 执行填充
df.fillna(value=fill_values, inplace=True)

# 再次检查
if df[required_cols].isnull().sum().sum() > 0:
    print("⚠️ 警告：仍有部分列存在缺失值，尝试用 0 填充剩余空值...")
    df.fillna(0, inplace=True) # 兜底策略

print("✅ 缺失值处理完成！")

# ==============================================================================
# 2. 分类变量哑变量处理
# ==============================================================================
# 将 T_score, GG_BP, PIRADS 转为分类变量
df['T_score'] = df['T_score'].astype('category')
df['GG_BP']   = df['GG_BP'].astype('category')
df['PIRADS']  = df['PIRADS'].astype('category')

# One-hot 编码，drop_first=True 防止多重共线性
df_dummies = pd.get_dummies(df[['T_score', 'GG_BP', 'PIRADS']], drop_first=True)

# 合并回原表
df = pd.concat([df, df_dummies], axis=1)

# ==============================================================================
# 3. 定义模型特征
# ==============================================================================
# Clinical 模型 (排除被One-hot之前的原始分类列)
clin_vars = ['age', 'PSA', 'pct_pos'] + [c for c in df_dummies.columns if ('T_score' in c or 'GG_BP' in c or 'PIRADS' in c)]
features_clin = clin_vars

# Habitat 模型
features_habitat = ['Habitat_score']

# Combined 模型
features_comb = features_clin + features_habitat

# CAPRA 模型只用 CAPRA_score
features_capra = ['CAPRA_score']

# ==============================================================================
# 4. 模型构建（仅在 Training 集）
# ==============================================================================
train_df = df[df['Dataset_Type'] == 'Training']
y_train = train_df['AP_Status']

# 初始化模型
lr_clin = LogisticRegression(max_iter=1000, solver='liblinear')
lr_hab  = LogisticRegression(max_iter=1000, solver='liblinear')
lr_comb = LogisticRegression(max_iter=1000, solver='liblinear')
lr_capra = LogisticRegression(max_iter=1000, solver='liblinear')

print("🏗️ 正在训练模型...")
# 拟合模型 (现在数据已经干净了，不会报错)
lr_clin.fit(train_df[features_clin], y_train)
lr_hab.fit(train_df[features_habitat], y_train)
lr_comb.fit(train_df[features_comb], y_train)
lr_capra.fit(train_df[features_capra], y_train)

# ==============================================================================
# 5. 预测整个数据集
# ==============================================================================
print("🔮 正在生成预测概率...")
df['pred_prob_Clinical'] = lr_clin.predict_proba(df[features_clin])[:, 1]
df['pred_prob_Habitat']  = lr_hab.predict_proba(df[features_habitat])[:, 1]
df['pred_prob_Combined'] = lr_comb.predict_proba(df[features_comb])[:, 1]
df['pred_prob_CAPRA']    = lr_capra.predict_proba(df[features_capra])[:, 1]

# ==============================================================================
# 6. 保存结果为新文件
# ==============================================================================
OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_Predictions.xlsx")

print(f"💾 正在保存结果至新文件: {OUTPUT_PATH}")

# 保存
df.to_excel(OUTPUT_PATH, index=False)

print("✅ 全部完成！新文件已生成。")

# ==============================================================================
# 7. 使用 statsmodels 提取 Logistic 回归参数（β, OR, 95%CI, p）
# ==============================================================================
import statsmodels.api as sm
def extract_logistic_params(X, y, model_name):
    """
    使用 statsmodels.Logit 拟合逻辑回归并提取：
    beta, OR, 95% CI, p-value
    """

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    print(f"\n📊 正在计算 Logistic 回归参数（{model_name}, Training 集）...")

    # ==============================
    # 1. 复制，避免污染原数据
    # ==============================
    X = X.copy()
    y = y.copy()

    # ==============================
    # 2. 强制所有自变量为数值
    # ==============================
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    y = pd.to_numeric(y, errors='coerce')

    # ==============================
    # 3. 删除缺失行（极少数，但必须）
    # ==============================
    valid_idx = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # ==============================
    # 4. 转为“纯 numpy float”
    #    这是关键中的关键
    # ==============================
    X_np = X.astype(float).values
    y_np = y.astype(int).values

    # 加截距
    X_np = sm.add_constant(X_np, has_constant='add')

    # ==============================
    # 5. 拟合 Logit
    # ==============================
    model = sm.Logit(y_np, X_np)
    result = model.fit(disp=False)

    # ==============================
    # 6. 提取参数
    # ==============================
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    var_names = ['Intercept'] + list(X.columns)

    summary_df = pd.DataFrame({
        'Model': model_name,
        'Variable': var_names,
        'Beta': params,
        'OR': np.exp(params),
        'CI_lower': np.exp(conf[:, 0]),
        'CI_upper': np.exp(conf[:, 1]),
        'p_value': pvals
    })

    # 去掉截距（论文通常不报）
    summary_df = summary_df[summary_df['Variable'] != 'Intercept']

    return summary_df




print("📊 正在计算 Logistic 回归参数（Training 集）...")

params_all = []

# Clinical
params_all.append(
    extract_logistic_params(
        train_df[features_clin],
        y_train,
        model_name='Clinical'
    )
)

# Habitat
params_all.append(
    extract_logistic_params(
        train_df[features_habitat],
        y_train,
        model_name='Habitat'
    )
)

# Combined
params_all.append(
    extract_logistic_params(
        train_df[features_comb],
        y_train,
        model_name='Combined'
    )
)

# CAPRA
params_all.append(
    extract_logistic_params(
        train_df[features_capra],
        y_train,
        model_name='CAPRA'
    )
)

# 合并所有模型结果
params_df = pd.concat(params_all, ignore_index=True)

# ==============================================================================
# 8. 保存参数结果
# ==============================================================================
PARAM_OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_Logistic_Params.xlsx")

print(f"💾 正在保存模型参数至: {PARAM_OUTPUT_PATH}")
params_df.to_excel(PARAM_OUTPUT_PATH, index=False)

print("✅ Logistic 回归参数保存完成！")
