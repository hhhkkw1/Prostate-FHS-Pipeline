# ==============================================================================
# Dataset_Type 分布均衡性分析
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.sandbox.stats.runs import runstest_1samp

# ==============================================================================
# 1. 读取数据
# ==============================================================================
FILE_PATH = r"I:\RP_multib\重新修改\副本ALL911改3.xlsx"

df = pd.read_excel(FILE_PATH)

# ==============================================================================
# 2. 检查 Dataset_Type
# ==============================================================================
if 'Dataset_Type' not in df.columns:
    raise ValueError("❌ 缺少 Dataset_Type 列")

# 行号
df['Row_Index'] = np.arange(len(df))

print("\n================ Dataset_Type Counts ================\n")
print(df['Dataset_Type'].value_counts())

# ==============================================================================
# 3. 编码为数字（便于可视化）
# ==============================================================================
mapping = {
    'Training': 0,
    'Internal_Val': 1,
    'External_Val': 2
}

df['Dataset_Code'] = df['Dataset_Type'].map(mapping)

# ==============================================================================
# 4. Runs Test（随机性检验）
# ==============================================================================
# 简化：
# Training vs 非Training
# ==============================================================================
binary_seq = (df['Dataset_Type'] == 'Training').astype(int)

z_stat, p_value = runstest_1samp(binary_seq)

print("\n================ Runs Test ================\n")
print(f"Z-statistic = {z_stat:.4f}")
print(f"P-value     = {p_value:.4e}")

if p_value < 0.05:
    print("❗ Dataset_Type 分布可能不是随机的（存在聚集）")
else:
    print("✅ Dataset_Type 分布接近随机")

# ==============================================================================
# 5. 计算滑动窗口比例
# ==============================================================================
window_size = 50

rolling_training = (
    (df['Dataset_Type'] == 'Training')
    .rolling(window_size)
    .mean()
)

rolling_internal = (
    (df['Dataset_Type'] == 'Internal_Val')
    .rolling(window_size)
    .mean()
)

rolling_external = (
    (df['Dataset_Type'] == 'External_Val')
    .rolling(window_size)
    .mean()
)

# ==============================================================================
# 6. 绘图
# ==============================================================================
plt.figure(figsize=(18, 5))

# ==============================================================================
# Plot 1: Dataset_Type沿行号分布
# ==============================================================================
plt.subplot(1, 2, 1)

sns.scatterplot(
    x='Row_Index',
    y='Dataset_Code',
    hue='Dataset_Type',
    data=df,
    s=40
)

plt.yticks(
    [0, 1, 2],
    ['Training', 'Internal', 'External']
)

plt.title('Dataset_Type Distribution Across Rows')

# ==============================================================================
# Plot 2: 滑动窗口比例
# ==============================================================================
plt.subplot(1, 2, 2)

plt.plot(
    df['Row_Index'],
    rolling_training,
    label='Training'
)

plt.plot(
    df['Row_Index'],
    rolling_internal,
    label='Internal_Val'
)

plt.plot(
    df['Row_Index'],
    rolling_external,
    label='External_Val'
)

plt.xlabel('Row Index')
plt.ylabel('Rolling Proportion')

plt.title(f'Rolling Distribution (window={window_size})')

plt.legend()

# ==============================================================================
# 保存
# ==============================================================================
OUTPUT_PATH = FILE_PATH.replace(
    ".xlsx",
    "_Dataset_Distribution.png"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(f"\n图像已保存:\n{OUTPUT_PATH}")

# ==============================================================================
# 7. 输出各Dataset区间
# ==============================================================================
print("\n================ Dataset Position Summary ================\n")

for dtype in df['Dataset_Type'].unique():

    idx = df[df['Dataset_Type'] == dtype].index

    print(f"{dtype}")
    print(f"  Count     : {len(idx)}")
    print(f"  First Row : {idx.min()}")
    print(f"  Last Row  : {idx.max()}")
    print("")