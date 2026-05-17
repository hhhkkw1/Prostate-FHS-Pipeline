import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter

# ================= 配置区 =================
FILE_PATH = r"I:/RP_multib/重新修改/副本ALL911改11.xlsx"
# 筛选出生存分析亚组的条件 (根据您的描述，是有BCR数据的386例)
# 假设筛选条件是 BCR 列不为空，或者有专门的标记
# 请根据实际情况修改筛选逻辑，这里假设所有 Dataset_Type 不为空且有 BCRmonth 的都是
TIME_COL = 'BCRmonth'
EVENT_COL = 'BCR'
GROUP_COL = 'Dataset_Type'

# ================= 计算函数 =================

def calculate_followup_stats(df, label="Total Cohort"):
    """
    计算标准的随访统计信息：
    1. Median Follow-up (Reverse KM method)
    2. Events n (%)
    3. 3-year & 5-year Survival Rate
    """
    T = df[TIME_COL]
    E = df[EVENT_COL]
    
    # 1. 计算中位随访时间 (Reverse Kaplan-Meier)
    # 核心技巧：将 Event 和 Censor 反转。
    # 原本：1=复发, 0=未复发
    # 逆向：1=未复发(作为事件), 0=复发(作为删失)
    kmf_rev = KaplanMeierFitter()
    kmf_rev.fit(T, event_observed=1-E) # 反转状态
    median_followup = kmf_rev.median_survival_time_
    
    # 获取随访时间的 IQR (直接基于原始时间数据的统计，或者基于生存函数的分布)
    # 通常临床论文汇报 IQR 是基于原始数据的分布（特别是针对 survivors），
    # 但为了严谨，我们这里计算所有人的时间分布 IQR
    q1 = np.percentile(T, 25)
    q3 = np.percentile(T, 75)
    
    # 2. 计算事件数
    n_events = E.sum()
    pct_events = n_events / len(df) * 100
    
    # 3. 计算 3年 和 5年 生存率 (Standard KM)
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)
    
    # 获取特定时间点的生存率 (Time以月为单位，3年=36月，5年=60月)
    # predict() 返回的是存活概率
    surv_3y = kmf.predict(36) * 100
    surv_5y = kmf.predict(60) * 100

    # 从 KM 的置信区间曲线中提取指定时间点的 95%CI
    # KM 为阶梯函数，这里取 <= t 的最近时点对应区间（与 KM 曲线读数一致）
    ci_df = kmf.confidence_interval_
    ci_times = ci_df.index.to_numpy(dtype=float)
    lower_col, upper_col = ci_df.columns[0], ci_df.columns[1]

    def _ci_at_time(t_months):
        pos = np.searchsorted(ci_times, t_months, side='right') - 1
        if pos < 0:
            pos = 0
        lower = ci_df.iloc[pos][lower_col] * 100
        upper = ci_df.iloc[pos][upper_col] * 100
        return lower, upper

    surv_3y_l, surv_3y_u = _ci_at_time(36)
    surv_5y_l, surv_5y_u = _ci_at_time(60)
    
    print(f"--- {label} (N={len(df)}) ---")
    print(f"Median Follow-up (Reverse KM): {median_followup:.1f} months")
    print(f"Follow-up IQR (Raw Data):      {q1:.1f} - {q3:.1f} months")
    print(f"Total Events (BCR):            {n_events} ({pct_events:.1f}%)")
    print(f"3-Year BCR-free Survival:      {surv_3y:.1f}% (95%CI {surv_3y_l:.1f}% - {surv_3y_u:.1f}%)")
    print(f"5-Year BCR-free Survival:      {surv_5y:.1f}% (95%CI {surv_5y_l:.1f}% - {surv_5y_u:.1f}%)")
    print("-" * 30)

# ================= 主程序 =================

# 1. 加载数据
df = pd.read_excel(FILE_PATH)

# 2. 筛选生存分析亚组 (N=386)
# 假设筛选逻辑是 BCRmonth 和 BCR 都不为空
df_surv = df.dropna(subset=[TIME_COL, EVENT_COL]).copy()
df_surv[TIME_COL] = pd.to_numeric(df_surv[TIME_COL], errors='coerce')
df_surv[EVENT_COL] = pd.to_numeric(df_surv[EVENT_COL], errors='coerce')

print(f"生存分析总队列人数: {len(df_surv)}")

# 3. 计算总体随访信息 (用于 Results 文字描述)
calculate_followup_stats(df_surv, "Total Prognostic Cohort")

# 4. (可选) 如果审稿人要求看 Training/Validation 分组的情况
groups = ['Training', 'Internal_Val', 'External_Val']
for g in groups:
    sub_df = df_surv[df_surv[GROUP_COL] == g]
    if len(sub_df) > 0:
        calculate_followup_stats(sub_df, f"Cohort: {g}")
