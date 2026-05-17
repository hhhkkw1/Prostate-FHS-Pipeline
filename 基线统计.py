import pandas as pd
import numpy as np
from scipy import stats
import warnings
import re

warnings.filterwarnings('ignore')

# ================= 配置区 =================
FILE_PATH = r"I:\RP_multib\重新修改\副本ALL911改11.xlsx"
OUTPUT_PATH = r"I:\RP_multib\重新修改\Table1_Baseline_Characteristics_New.xlsx"

# 定义数据集分组列名
GROUP_COL = 'Dataset_Type'
GROUPS_ORDER = ['Training', 'Internal_Val', 'External_Val']

# 1. 连续变量 (增加了 阳性针数, 阴性针数, pct_pos)
CONTINUOUS_VARS = [
    'age', 
    'PSA', 
    'Total_Tumor_vol',
    '阳性针数',
    '阴性针数',
    'pct_pos'
]

# 2. 分类变量 (字典格式: {Excel中的列名: 报告中显示的名称})
# 注意：移除了 Location, Upgrade, N Stage 等
CATEGORICAL_VARS = {
    'T_stage_grouped': 'Clinical T Stage',  # 这是我们代码生成的新列名
    'PIRADS': 'PI-RADS Score',
    'GG_BP': 'Biopsy GG',                   # 穿刺格里森
    'AP': 'Adverse Pathology',              # 新增
    'High_GG': 'High Grade GG',             # 新增
    'Lymph': 'Lymph Node Invasion' ,         # 新增
    
    # 如果还需要 EPE, SVI, Margin，请取消下面注释
    'EPE': 'Extraprostatic Extension',
    'SVI': 'Seminal Vesicle Invasion',
    'Margin': 'Positive Surgical Margin'
}

# ================= 工具函数 =================

def process_t_stage(val):
    """
    T分期清洗规则：
    1. T1 -> T1
    2. T3a, T3b, T4 -> ≥T3
    3. T2a, T2b, T2c -> 保持原样
    """
    s = str(val).strip()
    
    # 处理空值
    if s == 'nan' or s == '':
        return np.nan
        
    # 处理旧数据的数字编码 (假设 0=T2a, 1=T2b, 2=T2c, 3=T3a, 4=T3b)
    # 如果您的 Excel 已经是文本 (T2a)，这部分不会触发
    digit_map = {'0': 'T2a', '1': 'T2b', '2': 'T2c', '3': '≥T3', '4': '≥T3'}
    if s in digit_map:
        return digit_map[s]
    
    # 文本匹配规则
    s_upper = s.upper()
    if 'T1' in s_upper:
        return 'T1'
    elif 'T3' in s_upper or 'T4' in s_upper:
        return '≥T3'
    elif 'T2' in s_upper:
        # 尝试提取具体的 a, b, c
        if 'A' in s_upper: return 'T2a'
        if 'B' in s_upper: return 'T2b'
        if 'C' in s_upper: return 'T2c'
        return 'T2 NOS' # Not Otherwise Specified
    
    return s

def load_and_preprocess(path):
    print(f"正在加载数据: {path} ...")
    df = pd.read_excel(path)
    
    # 1. 列名统一映射 (确保 Excel 里的中文列名能对应上)
    col_map = {
        '年龄': 'Age', 
        'PI-RADS评分': 'PIRADS',
        '穿刺阳性针数': '阳性针数',  # 防止 Excel 列名不完全一致，做个防守
        '穿刺阴性针数': '阴性针数',
        '淋巴结': 'Lymph'
    }
    # 仅重命名存在的列
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    
    # 2. 处理 T 分期 (逻辑升级)
    # 优先使用 'T_new'，如果没有则找 'T_score'
    raw_t_col = 'T_new' if 'T_new' in df.columns else ('T_score' if 'T_score' in df.columns else None)
    
    if raw_t_col:
        print(f"正在处理 T 分期，源列: {raw_t_col} ...")
        df['T_stage_grouped'] = df[raw_t_col].apply(process_t_stage)
    else:
        print("⚠️ 警告: 未找到 T 分期相关列 (T_new 或 T_score)")
        df['T_stage_grouped'] = np.nan

    # 3. 计算 Total Volume (如果不存在)
    if 'Total_Tumor_Vol' not in df.columns:
        vol_cols = [c for c in df.columns if 'Volume_ml' in c and 'Cluster' in c]
        if vol_cols:
            df['Total_Tumor_Vol'] = df[vol_cols].sum(axis=1)

    # === 【核心修复】强制处理所有分类变量为字符串 ===
    print("正在标准化分类变量格式...")
    for var in CATEGORICAL_VARS.keys():
        if var in df.columns:
            # 1. 转为字符串
            df[var] = df[var].astype(str)
            # 2. 去除空值字符串 'nan'
            df[var] = df[var].replace('nan', np.nan)
            # 3. 去除浮点数后缀 (例如 '1.0' -> '1', '0.0' -> '0')
            df[var] = df[var].apply(lambda x: str(x).replace('.0', '') if pd.notnull(x) else x)
            
    # === 确保连续变量是数字 ===
    for var in CONTINUOUS_VARS:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')

    return df

def analyze_continuous(df, var, groups):
    """分析连续变量 (ANOVA 或 Kruskal-Wallis)"""
    group_data = [df[df[GROUP_COL] == g][var].dropna() for g in groups]
    
    # 检查是否有数据
    if any(len(d) == 0 for d in group_data):
        return {g: 'ND' for g in groups}, np.nan, "No Data"

    # 正态性检验
    is_normal = True
    for d in group_data:
        if len(d) < 3: continue 
        stat, p = stats.shapiro(d)
        if p < 0.05:
            is_normal = False
            break
    
    row_res = {}
    p_val = np.nan
    
    try:
        if is_normal:
            f_stat, p_val = stats.f_oneway(*group_data)
            test_name = "ANOVA"
            for i, g in enumerate(groups):
                mean = group_data[i].mean()
                std = group_data[i].std()
                row_res[g] = f"{mean:.2f} ± {std:.2f}"
        else:
            stat, p_val = stats.kruskal(*group_data)
            test_name = "Kruskal-Wallis"
            for i, g in enumerate(groups):
                med = group_data[i].median()
                q1 = group_data[i].quantile(0.25)
                q3 = group_data[i].quantile(0.75)
                row_res[g] = f"{med:.2f} ({q1:.2f}-{q3:.2f})"
    except:
        test_name = "Error"

    return row_res, p_val, test_name

def analyze_categorical(df, var, groups):
    """分析分类变量 (Chi-square)"""
    # 选取子集
    sub_df = df[df[GROUP_COL].isin(groups)].copy()
    
    # 获取排序后的唯一值 (排除空值)
    unique_vals = sorted([x for x in sub_df[var].unique() if pd.notnull(x)])
    
    # 如果没有值，直接返回
    if not unique_vals:
        return [], np.nan, "No Data"

    # 创建交叉表
    crosstab = pd.crosstab(sub_df[var], sub_df[GROUP_COL])
    crosstab = crosstab.reindex(columns=groups, fill_value=0)
    
    # 检验
    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(crosstab)
        test_name = "Chi-square"
    except:
        p_val = np.nan
        test_name = "Error"
        
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

def generate_table1():
    df = load_and_preprocess(FILE_PATH)
    
    results = []
    
    # 表头 N
    header = {'Variable': 'Total No. of Patients'}
    for g in GROUPS_ORDER:
        n = len(df[df[GROUP_COL] == g])
        header[g] = f"N={n}"
    header['P Value'] = ''
    header['Test Method'] = ''
    results.append(header)
    
    # 1. 连续变量
    for var in CONTINUOUS_VARS:
        if var not in df.columns:
            print(f"⚠️ 警告: 找不到连续变量 {var}")
            continue
        row_data, p, method = analyze_continuous(df, var, GROUPS_ORDER)
        p_str = "<0.001" if (pd.notnull(p) and p < 0.001) else (f"{p:.3f}" if pd.notnull(p) else "-")
        
        full_row = {'Variable': var}
        full_row.update(row_data)
        full_row['P Value'] = p_str
        full_row['Test Method'] = method
        results.append(full_row)
        
    # 2. 分类变量
    for var, display_name in CATEGORICAL_VARS.items():
        if var not in df.columns:
            print(f"⚠️ 警告: 找不到分类变量 {var} (对应显示名称: {display_name})")
            continue
            
        sub_rows, p, method = analyze_categorical(df, var, GROUPS_ORDER)
        p_str = "<0.001" if (pd.notnull(p) and p < 0.001) else (f"{p:.3f}" if pd.notnull(p) else "-")
        
        # 变量头
        title_row = {'Variable': display_name, 'P Value': p_str, 'Test Method': method}
        for g in GROUPS_ORDER: title_row[g] = ''
        results.append(title_row)
        
        # 类别行
        results.extend(sub_rows)

    # 导出
    res_df = pd.DataFrame(results)
    cols = ['Variable'] + GROUPS_ORDER + ['P Value', 'Test Method']
    res_df = res_df[cols]
    
    print("\n📊 修正后的 Table 1 预览:")
    print(res_df.head(20).to_string(index=False))
    
    res_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n✅ 表格已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_table1()