import pandas as pd
import numpy as np
from scipy import stats
import warnings
import re

warnings.filterwarnings('ignore')

# ================= 配置区 =================
FILE_PATH = r"I:\RP_multib\副本ALL911.xlsx"
OUTPUT_PATH = r"I:\RP_multib\Supplementary_Table_Survival_Cohort.xlsx"

# 定义数据集分组列名
GROUP_COL = 'Dataset_Type'
GROUPS_ORDER = ['Training', 'Internal_Val', 'External_Val']

# 1. 连续变量 (在原有基础上增加 'BCRmonth')
CONTINUOUS_VARS = [
    'age', 
    'PSA', 
    'Total_Tumor_vol',
    '阳性针数',
    '阴性针数',
    'pct_pos',
    'BCRmonth'  # <--- 新增：随访时间
]

# 2. 分类变量 (在原有基础上增加 'BCR')
CATEGORICAL_VARS = {
    'T_stage_grouped': 'Clinical T Stage',
    'PIRADS': 'PI-RADS Score',
    'GG_BP': 'Biopsy GG',
    'AP': 'Adverse Pathology',
    'High_GG': 'High Grade GG',
    'Lymph': 'Lymph Node Invasion',
    'EPE': 'Extraprostatic Extension',
    'SVI': 'Seminal Vesicle Invasion',
    'Margin': 'Positive Surgical Margin',
    'BCR': 'Biochemical Recurrence (Event)' # <--- 新增：复发事件
}

# ================= 工具函数 (保持不变) =================

def process_t_stage(val):
    """T分期清洗规则"""
    s = str(val).strip()
    if s == 'nan' or s == '': return np.nan
    digit_map = {'0': 'T2a', '1': 'T2b', '2': 'T2c', '3': '≥T3', '4': '≥T3'}
    if s in digit_map: return digit_map[s]
    s_upper = s.upper()
    if 'T1' in s_upper: return 'T1'
    elif 'T3' in s_upper or 'T4' in s_upper: return '≥T3'
    elif 'T2' in s_upper:
        if 'A' in s_upper: return 'T2a'
        if 'B' in s_upper: return 'T2b'
        if 'C' in s_upper: return 'T2c'
        return 'T2 NOS'
    return s

def load_and_preprocess(path):
    print(f"正在加载数据: {path} ...")
    df = pd.read_excel(path)
    
    # 1. 筛选生存分析亚组 (核心步骤)
    # 逻辑：BCR 和 BCRmonth 都不为空的病人
    initial_len = len(df)
    df = df.dropna(subset=['BCR', 'BCRmonth']).copy()
    print(f"🔍 筛选生存队列: 从 {initial_len} 例筛选至 {len(df)} 例")

    # 2. 列名映射
    col_map = {
        '年龄': 'Age', 'PI-RADS评分': 'PIRADS',
        '穿刺阳性针数': '阳性针数', '穿刺阴性针数': '阴性针数', '淋巴结': 'Lymph'
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    
    # 3. 处理 T 分期
    raw_t_col = 'T_new' if 'T_new' in df.columns else ('T_score' if 'T_score' in df.columns else None)
    if raw_t_col:
        df['T_stage_grouped'] = df[raw_t_col].apply(process_t_stage)
    else:
        df['T_stage_grouped'] = np.nan

    # 4. 计算 Total Volume
    if 'Total_Tumor_Vol' not in df.columns:
        vol_cols = [c for c in df.columns if 'Volume_ml' in c and 'Cluster' in c]
        if vol_cols: df['Total_Tumor_Vol'] = df[vol_cols].sum(axis=1)

    # 5. 标准化分类变量 & 确保连续变量为数字
    for var in CATEGORICAL_VARS.keys():
        if var in df.columns:
            df[var] = df[var].astype(str).replace('nan', np.nan).apply(lambda x: str(x).replace('.0', '') if pd.notnull(x) else x)
            
    for var in CONTINUOUS_VARS:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')

    return df

def analyze_continuous(df, var, groups):
    """分析连续变量"""
    group_data = [df[df[GROUP_COL] == g][var].dropna() for g in groups]
    if any(len(d) == 0 for d in group_data): return {g: 'ND' for g in groups}, np.nan, "No Data"

    # 正态性检验
    is_normal = True
    for d in group_data:
        if len(d) < 3: continue 
        stat, p = stats.shapiro(d)
        if p < 0.05: is_normal = False; break
    
    row_res = {}
    p_val = np.nan
    try:
        if is_normal:
            f_stat, p_val = stats.f_oneway(*group_data)
            test_name = "ANOVA"
            for i, g in enumerate(groups):
                row_res[g] = f"{group_data[i].mean():.2f} ± {group_data[i].std():.2f}"
        else:
            stat, p_val = stats.kruskal(*group_data)
            test_name = "Kruskal-Wallis"
            for i, g in enumerate(groups):
                row_res[g] = f"{group_data[i].median():.2f} ({group_data[i].quantile(0.25):.2f}-{group_data[i].quantile(0.75):.2f})"
    except: test_name = "Error"
    return row_res, p_val, test_name

def analyze_categorical(df, var, groups):
    """分析分类变量"""
    sub_df = df[df[GROUP_COL].isin(groups)].copy()
    unique_vals = sorted([x for x in sub_df[var].unique() if pd.notnull(x)])
    if not unique_vals: return [], np.nan, "No Data"

    crosstab = pd.crosstab(sub_df[var], sub_df[GROUP_COL]).reindex(columns=groups, fill_value=0)
    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(crosstab)
        test_name = "Chi-square"
    except: p_val = np.nan; test_name = "Error"
        
    rows = []
    group_counts = sub_df[GROUP_COL].value_counts()
    for val in unique_vals:
        row_dict = {'Variable': f"  {val}"}
        for g in groups:
            count = crosstab.loc[val, g] if val in crosstab.index else 0
            total = group_counts.get(g, 0)
            pct = (count / total * 100) if total > 0 else 0
            row_dict[g] = f"{count} ({pct:.1f}%)"
        rows.append(row_dict)
    return rows, p_val, test_name

# ================= 主程序 =================

def generate_supp_table():
    df = load_and_preprocess(FILE_PATH)
    results = []
    
    # 表头
    header = {'Variable': 'Total (Survival Cohort)'}
    for g in GROUPS_ORDER:
        n = len(df[df[GROUP_COL] == g])
        header[g] = f"N={n}"
    header['P Value'] = ''
    header['Test Method'] = ''
    results.append(header)
    
    # 1. 连续变量
    for var in CONTINUOUS_VARS:
        if var not in df.columns: continue
        # 如果是 BCRmonth，显示名称改一下
        display_var = "Follow-up Duration (months)" if var == 'BCRmonth' else var
        
        row_data, p, method = analyze_continuous(df, var, GROUPS_ORDER)
        p_str = "<0.001" if (pd.notnull(p) and p < 0.001) else (f"{p:.3f}" if pd.notnull(p) else "-")
        full_row = {'Variable': display_var}
        full_row.update(row_data)
        full_row['P Value'] = p_str
        full_row['Test Method'] = method
        results.append(full_row)
        
    # 2. 分类变量
    for var, display_name in CATEGORICAL_VARS.items():
        if var not in df.columns: continue
        sub_rows, p, method = analyze_categorical(df, var, GROUPS_ORDER)
        p_str = "<0.001" if (pd.notnull(p) and p < 0.001) else (f"{p:.3f}" if pd.notnull(p) else "-")
        
        title_row = {'Variable': display_name, 'P Value': p_str, 'Test Method': method}
        for g in GROUPS_ORDER: title_row[g] = ''
        results.append(title_row)
        results.extend(sub_rows)

    # 导出
    res_df = pd.DataFrame(results)
    cols = ['Variable'] + GROUPS_ORDER + ['P Value', 'Test Method']
    res_df = res_df[cols]
    
    print("\n📊 Survival Cohort Table 预览:")
    print(res_df.head(20).to_string(index=False))
    
    res_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n✅ 附录表格已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_supp_table()