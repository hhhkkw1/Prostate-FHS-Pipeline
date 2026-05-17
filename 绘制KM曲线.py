import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import os

# ==============================================================================
# 0. 配置与环境 (Publication-Ready Style)
# ==============================================================================
# 全局字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5      # 坐标轴线宽
plt.rcParams['xtick.major.width'] = 1.5   # X轴刻度线宽
plt.rcParams['ytick.major.width'] = 1.5   # Y轴刻度线宽
plt.rcParams['xtick.major.size'] = 6      # X轴刻度长度
plt.rcParams['ytick.major.size'] = 6      # Y轴刻度长度

FILE_PATH = "I:/RP_multib/重新修改/副本ALL911改11.xlsx"
RISK_TABLE_TIMEPOINTS = [0, 24, 48, 72, 96, 120]

# ==============================================================================
# 1. 数据读取与清洗
# ==============================================================================
print(f"📂 正在读取数据: {FILE_PATH}")
df = pd.read_excel(FILE_PATH)

df_surv = df.dropna(subset=['BCR', 'BCRmonth']).copy()
df_surv['BCRmonth'] = pd.to_numeric(df_surv['BCRmonth'], errors='coerce')
df_surv['BCR'] = pd.to_numeric(df_surv['BCR'], errors='coerce')

# ==============================================================================
# 2. 寻找最佳截断值 (Optimal Cutoff) - 仅基于 Training 集
# ==============================================================================
print("\n🔍 正在 Training 集中寻找 Habitat_score 的最佳截断值...")

train_df = df_surv[df_surv['Dataset_Type'] == 'Training']
candidate_cuts = np.percentile(train_df['Habitat_score'], np.arange(10, 91, 1))
best_p = 1.0
best_cut = train_df['Habitat_score'].median()

for cut in candidate_cuts:
    group_high = train_df[train_df['Habitat_score'] > cut]
    group_low = train_df[train_df['Habitat_score'] <= cut]
    
    if len(group_high) < 10 or len(group_low) < 10: continue
        
    # 使用 .values 防止索引对齐报错
    results = logrank_test(
        durations_A=group_high['BCRmonth'].values, event_observed_A=group_high['BCR'].values,
        durations_B=group_low['BCRmonth'].values, event_observed_B=group_low['BCR'].values
    )
    
    if results.p_value < best_p:
        best_p = results.p_value
        best_cut = cut

print(f"✅ 最佳截断值 (Cut-off): {best_cut:.4f}")

# ==============================================================================
# 3. 绘制美化版 KM 曲线 (Training, Internal, External)
# ==============================================================================
datasets = ['Training', 'Internal_Val', 'External_Val']
dataset_labels = {'Training': 'Training', 'Internal_Val': 'Internal_Val', 'External_Val': 'External_Val'}

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for i, dtype in enumerate(datasets):
    ax = axes[i]
    data_subset = df_surv[df_surv['Dataset_Type'] == dtype]
    
    # 分组
    high_risk = data_subset[data_subset['Habitat_score'] > best_cut]
    low_risk  = data_subset[data_subset['Habitat_score'] <= best_cut]
    
    # 计算 P 值
    p_text = "P = N/A"
    if len(high_risk) > 0 and len(low_risk) > 0:
        lr_result = logrank_test(
            durations_A=high_risk['BCRmonth'].values, event_observed_A=high_risk['BCR'].values,
            durations_B=low_risk['BCRmonth'].values, event_observed_B=low_risk['BCR'].values
        )
        p_val = lr_result.p_value
        p_text = f"P < 0.001" if p_val < 0.001 else f"P = {p_val:.4f}"

    # --- 绘图核心 ---
    # Low Risk (Blue)
    kmf_low = KaplanMeierFitter()
    kmf_low.fit(low_risk['BCRmonth'], low_risk['BCR'], label=f'Low Risk (N={len(low_risk)})')
    kmf_low.plot_survival_function(ax=ax, color='#4DBBD5', linewidth=2.5, show_censors=True, ci_show=True, ci_alpha=0.1)

    # High Risk (Red)
    kmf_high = KaplanMeierFitter()
    kmf_high.fit(high_risk['BCRmonth'], high_risk['BCR'], label=f'High Risk (N={len(high_risk)})')
    kmf_high.plot_survival_function(ax=ax, color='#E64B35', linewidth=2.5, show_censors=True, ci_show=True, ci_alpha=0.1)

    # --- 深度美化 ---
    # 1. 标题与轴标签
    ax.set_title(dataset_labels[dtype], fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Time (Months)", fontsize=14, fontweight='bold')
    
    if i == 0:
        ax.set_ylabel("BCR-free Survival Probability", fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel("") # 只在第一张图显示Y轴标签
        ax.set_yticklabels([]) # 隐藏Y轴刻度标签

    # 2. 坐标轴范围与样式
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(df_surv['BCRmonth']) * 1.05)
    ax.set_xticks(RISK_TABLE_TIMEPOINTS)
    
    # 去除上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 去除网格 (Seaborn 默认有，这里强制关掉)
    ax.grid(False)

    # 3. Log-rank P 值 (右上角)
    ax.text(0.95, 0.90, f"Log-rank\n{p_text}", 
            transform=ax.transAxes, ha='right', va='top', fontsize=13, fontweight='bold')

    # 4. 图例与 Cut-off (左下角)
    # 自定义图例位置和样式
    legend = ax.legend(loc='lower left', fontsize=11, frameon=False, title=f"FHS Cut-off: {best_cut:.3f}")
    legend.get_title().set_fontsize(11) # 设置图例标题大小
    legend.get_title().set_fontweight('bold')

    # 5. Number at risk（并显示 censored）
    add_at_risk_counts(
        kmf_low, kmf_high,
        ax=ax,
        rows_to_show=['At risk', 'Censored'],
        xticks=RISK_TABLE_TIMEPOINTS,
        fontsize=12
    )
plt.tight_layout(rect=[0, 0.14, 1, 1])
plt.subplots_adjust(bottom=0.31, wspace=0.25)
save_path = FILE_PATH.replace(".xlsx", "_KM_Curves_Final1.tiff")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"📉 最终发表级 KM 曲线已保存至: {save_path}")
