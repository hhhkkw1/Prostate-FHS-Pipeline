# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import os

# ==============================================================================
# 配置区域
# ==============================================================================
FILE_PATH = "I:/RP_multib/重新修改/副本ALL911改11.xlsx"
OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_with_CAPRA.xlsx")

# ==============================================================================
# 辅助函数
# ==============================================================================
def calculate_psa_points(psa):
    if pd.isna(psa): return 0
    if psa < 6: return 0
    elif psa < 10: return 1
    elif psa < 20: return 2
    elif psa < 30: return 3
    else: return 4

def calculate_gs_points(gs_str):
    if pd.isna(gs_str): return 0
    nums = [int(x) for x in re.findall(r'\d+', str(gs_str))]
    if len(nums) < 2: return 0
    primary, secondary = nums[0], nums[1]
    if primary >= 4: return 3
    elif secondary >= 4: return 1
    else: return 0

def main():
    print(f"📂 [步骤1] 正在读取数据: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print(f"❌ 文件不存在！")
        return
    df = pd.read_excel(FILE_PATH)
    print(f"   -> 读取成功，共 {len(df)} 行数据")

    # -----------------------------------------------------------
    # 列名检查
    # -----------------------------------------------------------
    print("🔍 [步骤2] 检查必要列名...")
    required_cols = ['age', 'PSA', 'T_score', '阳性针数', '阴性针数']
    
    # 自动修正 GS 列名
    gs_col = None
    if 'GS_PB' in df.columns: gs_col = 'GS_PB'
    elif 'GS_BP' in df.columns: gs_col = 'GS_BP'
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ 缺少列: {missing}")
        return
    if not gs_col:
        print("❌ 缺少 Gleason Score 列 (GS_PB 或 GS_BP)")
        return
    print("   -> 所有必要列均存在。")

    # -----------------------------------------------------------
    # 分项计算 (每一步都打印确认)
    # -----------------------------------------------------------
    print("🧮 [步骤3] 开始分项计算...")

    # 1. Age
    df['score_age'] = (df['age'] >= 50).astype(int)
    print("   -> Age 分数计算完毕")

    # 2. PSA
    df['score_psa'] = df['PSA'].apply(calculate_psa_points)
    print("   -> PSA 分数计算完毕")

    # 3. GS
    df['score_gs'] = df[gs_col].apply(calculate_gs_points)
    print("   -> GS 分数计算完毕")

    # 4. T-score
    # 填充空值防止报错
    df['T_score'] = df['T_score'].fillna(0)
    df['score_t'] = (df['T_score'] >= 4).astype(int)
    print("   -> T-score 分数计算完毕")

    # 5. % Positive Cores
    # 先处理分母为0的情况
    total_cores = df['阳性针数'] + df['阴性针数']
    df['pct_pos'] = df['阳性针数'] / total_cores
    df['pct_pos'] = df['pct_pos'].fillna(0) # 填充空值
    df['score_pct'] = (df['pct_pos'] >= 0.34).astype(int)
    print("   -> 阳性针数分数计算完毕")

    # -----------------------------------------------------------
    # 汇总计算 (最关键的一步)
    # -----------------------------------------------------------
    print("∑  [步骤4] 正在汇总 CAPRA 总分...")
    
    # 强制检查所有子分数是否存在
    sub_scores = ['score_age', 'score_psa', 'score_gs', 'score_t', 'score_pct']
    for col in sub_scores:
        if col not in df.columns:
            raise ValueError(f"❌ 严重错误: 子分数 {col} 计算失败，未出现在表格中！")
            
    # 执行加法
    df['CAPRA_score'] = (
        df['score_age'] + 
        df['score_psa'] + 
        df['score_gs'] + 
        df['score_t'] + 
        df['score_pct']
    )
    
    # 再次确认
    if 'CAPRA_score' not in df.columns:
        raise KeyError("❌ 严重错误: CAPRA_score 列创建失败！")
    else:
        print("   -> CAPRA_score 列创建成功！")

    # -----------------------------------------------------------
    # 统计与保存
    # -----------------------------------------------------------
    print("\n📊 [步骤5] CAPRA 评分统计分布:")
    print(df['CAPRA_score'].value_counts().sort_index())
    
    print(f"\n💾 [步骤6] 正在保存至: {OUTPUT_PATH}")
    df.to_excel(OUTPUT_PATH, index=False)
    print("✅ 全部完成！")

if __name__ == "__main__":
    main()