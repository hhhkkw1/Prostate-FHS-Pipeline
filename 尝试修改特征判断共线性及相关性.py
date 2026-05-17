import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 读取数据
# =========================

file_path = r"I:\RP_multib\重新修改\副本ALL911改11.xlsx"

df = pd.read_excel(file_path)

# =========================
# 特征与结局
# =========================

features = [
    'h1_Sphericity',
    'h1_Frag',
    'h0_Ratio',
    'h2_Sphericity',
    'h2_Ratio',
    'h3_f_Std',
    'h3_f_Skew'
]

target = 'AP_Status'

# 删除缺失值
data = df[features + [target]].dropna()

# =========================
# 1. 描述统计
# =========================

desc = data[features].describe().T

print("\n===== 描述统计 =====")
print(desc)

# =========================
# 2. 与结局相关性
# Spearman correlation
# =========================

print("\n===== 与结局相关性 =====")

corr_results = []

for feat in features:
    corr, p = spearmanr(data[feat], data[target])

    corr_results.append({
        'Feature': feat,
        'Spearman_r': corr,
        'P_value': p
    })

corr_df = pd.DataFrame(corr_results)

print(corr_df)

# 保存
#corr_df.to_excel(r"I:\RP_multib\重新修改\feature_target_correlation.xlsx", index=False)

# =========================
# 3. 特征间相关性热图
# =========================

corr_matrix = data[features].corr(method='spearman')

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    square=True
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

#plt.savefig(r"I:\RP_multib\重新修改\feature_correlation_heatmap.png", dpi=300)
plt.show()

# =========================
# 4. Pairplot
# =========================


# =========================
# 5. VIF 共线性分析
# =========================

X = data[features]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

vif_data = pd.DataFrame()

vif_data["Feature"] = features

vif_data["VIF"] = [
    variance_inflation_factor(X_scaled, i)
    for i in range(len(features))
]

print("\n===== VIF =====")
print(vif_data)

#vif_data.to_excel(r"I:\RP_multib\重新修改\vif_results.xlsx", index=False)

# =========================
# 6. PCA分析
# =========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    target: data[target].values
})

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue=target
)

plt.title("PCA of Selected Features")

plt.tight_layout()

#plt.savefig(r"I:\RP_multib\重新修改\pca_scatter.png", dpi=300)
plt.show()

# explained variance
print("\n===== PCA Explained Variance =====")
print(pca.explained_variance_ratio_)


print("\n分析完成！")