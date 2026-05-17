import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# ==============================================================================
# 0. 数据读取与预处理
# ==============================================================================
FILE_PATH = "I:\RP_multib\重新修改\副本ALL911改11.xlsx"
print(f"📂 正在读取数据: {FILE_PATH}")
df = pd.read_excel(FILE_PATH)

# 1. 筛选有 BCR 结局的病人
df_surv = df.dropna(subset=['BCR', 'BCRmonth']).copy()

# 2. 强制数值化 (Continuous)
# 这一步非常关键，确保 T_score 和 GG_BP 是按数值处理，而不是分类
cols_to_numeric = ['BCRmonth', 'BCR', 'age', 'PSA', 'pct_pos', 'Habitat_score', 'T_score', 'GG_BP', 'PIRADS']
for col in cols_to_numeric:
    df_surv[col] = pd.to_numeric(df_surv[col], errors='coerce')

# 3. 仅使用 Training 集进行筛选 (标准做法)
df_train = df_surv[df_surv['Dataset_Type'] == 'Training'].copy()

# 填充缺失值 (以防万一)
medians = df_train[cols_to_numeric].median()
df_train.fillna(medians, inplace=True)

print(f"📊 纳入单因素分析样本数 (Training): {len(df_train)} 例")
print(f"   发生 BCR 事件数: {int(df_train['BCR'].sum())} 例")

# ==============================================================================
# 1. 单因素 Cox 回归分析 (Univariate Cox Regression)
# ==============================================================================
# 定义待分析变量列表
uni_vars = ['age', 'PSA', 'pct_pos', 'T_score', 'GG_BP', 'Habitat_score',"CAPRA_score", 'PIRADS']

results = []

print("\n🚀 开始单因素 Cox 回归分析...")
print("-" * 80)
print(f"{'Variable':<15} | {'HR (Exp Coef)':<15} | {'95% CI':<20} | {'P-value':<10} | {'Sig.'}")
print("-" * 80)

for var in uni_vars:
    try:
        cph = CoxPHFitter()
        # 提取单个变量 + 时间 + 状态
        data_subset = df_train[[var, 'BCRmonth', 'BCR']]
        
        # 拟合
        cph.fit(data_subset, duration_col='BCRmonth', event_col='BCR')
        
        # 提取结果
        summary = cph.summary
        hr = summary.loc[var, 'exp(coef)']
        lower = summary.loc[var, 'exp(coef) lower 95%']
        upper = summary.loc[var, 'exp(coef) upper 95%']
        p = summary.loc[var, 'p']
        
        # 显著性标记
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
        
        print(f"{var:<15} | {hr:.4f}          | {lower:.3f} - {upper:.3f}    | {p:.4f}     | {sig}")
        
        results.append({
            'Variable': var,
            'HR': hr,
            'CI_Lower': lower,
            'CI_Upper': upper,
            'P_value': p
        })
        
    except Exception as e:
        print(f"{var:<15} | ❌ 计算失败: {str(e)}")

print("-" * 80)
print("注: HR > 1 代表风险增加; HR < 1 代表风险降低")