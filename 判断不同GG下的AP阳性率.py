# ==============================================================================
# GG_BP 与 AP_Status 关系分析
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. 读取数据
# ==============================================================================
FILE_PATH = r"I:\RP_multib\重新修改\副本ALL911.xlsx"

df = pd.read_excel(FILE_PATH)

# ==============================================================================
# 2. 数据清洗
# ==============================================================================
df['GG_BP'] = pd.to_numeric(df['GG_BP'], errors='coerce')
df['AP_Status'] = pd.to_numeric(df['AP_Status'], errors='coerce')

df = df.dropna(subset=['GG_BP', 'AP_Status'])

df['AP_Status'] = df['AP_Status'].astype(int)

print("数据量:", len(df))

# ==============================================================================
# 3. 统计每个GG等级的AP阳性率
# ==============================================================================
summary = df.groupby('GG_BP').agg(
    Total=('AP_Status', 'count'),
    Positive=('AP_Status', 'sum')
).reset_index()

summary['Positive_Rate'] = summary['Positive'] / summary['Total']

print("\n================ GG_BP 与 AP阳性率 ================\n")
print(summary)

# ==============================================================================
# 4. Spearman相关性
# ==============================================================================
rho, p = spearmanr(df['GG_BP'], df['AP_Status'])

print("\n================ Spearman相关性 ================\n")
print(f"Spearman rho = {rho:.4f}")
print(f"P-value       = {p:.4e}")

# ==============================================================================
# 5. Logistic回归拟合
# ==============================================================================
X = df[['GG_BP']]
y = df['AP_Status']

lr = LogisticRegression()

lr.fit(X, y)

beta = lr.coef_[0][0]
intercept = lr.intercept_[0]

OR = np.exp(beta)

print("\n================ Logistic回归 ================\n")
print(f"Beta = {beta:.4f}")
print(f"OR   = {OR:.4f}")

# ==============================================================================
# 6. 生成预测曲线
# ==============================================================================
x_range = np.linspace(
    df['GG_BP'].min(),
    df['GG_BP'].max(),
    200
).reshape(-1, 1)

pred_prob = lr.predict_proba(x_range)[:, 1]

# ==============================================================================
# 7. 绘图
# ==============================================================================
plt.figure(figsize=(18, 5))

# ==============================================================================
# Plot 1: 阳性率柱状图
# ==============================================================================
plt.subplot(1, 3, 1)

sns.barplot(
    x='GG_BP',
    y='Positive_Rate',
    data=summary
)

plt.title('AP Positive Rate by GG_BP')
plt.ylabel('AP Positive Rate')
plt.ylim(0, 1)

for i, row in summary.iterrows():
    plt.text(
        i,
        row['Positive_Rate'] + 0.02,
        f"{row['Positive_Rate']:.2f}",
        ha='center'
    )

# ==============================================================================
# Plot 2: 散点 + 趋势
# ==============================================================================
plt.subplot(1, 3, 2)

sns.stripplot(
    x='GG_BP',
    y='AP_Status',
    data=df,
    jitter=0.2,
    alpha=0.4
)

sns.pointplot(
    x='GG_BP',
    y='AP_Status',
    data=df,
    estimator=np.mean,
    errorbar=('ci', 95),
    color='red'
)

plt.title('Distribution of AP_Status across GG_BP')

# ==============================================================================
# Plot 3: Logistic拟合曲线
# ==============================================================================
plt.subplot(1, 3, 3)

plt.scatter(
    df['GG_BP'],
    df['AP_Status'],
    alpha=0.2
)

plt.plot(
    x_range,
    pred_prob,
    linewidth=3
)

plt.xlabel('GG_BP')
plt.ylabel('Predicted Probability')

plt.title('Logistic Curve')

# ==============================================================================
# 保存
# ==============================================================================
OUTPUT_FIG = FILE_PATH.replace(".xlsx", "_GG_AP_relationship.png")

plt.tight_layout()

plt.savefig(
    OUTPUT_FIG,
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(f"\n图像已保存:\n{OUTPUT_FIG}")