import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ====================================================
# 读取数据
# ====================================================

file_path = r"I:\RP_multib\重新修改\副本ALL911改3.xlsx"

df = pd.read_excel(file_path)

# ====================================================
# 选择预测概率列
# ====================================================

pred_cols = [
    'pred_prob_Clinical',
    'pred_prob_Habitat',
    'pred_prob_Combined',
    'pred_prob_CAPRA'
]

data = df[pred_cols].dropna()

# ====================================================
# 1. Pearson相关性
# ====================================================

pearson_corr = data.corr(method='pearson')

print("\n===== Pearson Correlation =====")
print(pearson_corr)

# 保存
pearson_corr.to_excel(
    r"I:\RP_multib\重新修改\pearson_correlation.xlsx"
)

# ====================================================
# 2. Spearman相关性
# ====================================================

spearman_corr = data.corr(method='spearman')

print("\n===== Spearman Correlation =====")
print(spearman_corr)

spearman_corr.to_excel(
    r"I:\RP_multib\重新修改\spearman_correlation.xlsx"
)

# ====================================================
# 3. 热图
# ====================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    pearson_corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    square=True
)

plt.title("Pearson Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    r"I:\RP_multib\重新修改\prediction_correlation_heatmap.png",
    dpi=300
)

plt.show()

# ====================================================
# 4. Pairplot
# ====================================================

sns.pairplot(data)

plt.savefig(
    r"I:\RP_multib\重新修改\prediction_pairplot.png",
    dpi=300
)

plt.show()

# ====================================================
# 5. VIF 共线性分析
# ====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(data)

vif_df = pd.DataFrame()

vif_df['Variable'] = pred_cols

vif_df['VIF'] = [
    variance_inflation_factor(X_scaled, i)
    for i in range(len(pred_cols))
]

print("\n===== VIF =====")
print(vif_df)

vif_df.to_excel(
    r"I:\RP_multib\重新修改\prediction_vif.xlsx",
    index=False
)

# ====================================================
# 6. PCA分析
# ====================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:,0],
    'PC2': X_pca[:,1]
})

plt.figure(figsize=(7,6))

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2'
)

plt.title("PCA of Prediction Probabilities")

plt.tight_layout()

plt.savefig(
    r"I:\RP_multib\重新修改\prediction_pca.png",
    dpi=300
)

plt.show()

print("\n===== PCA Explained Variance =====")
print(pca.explained_variance_ratio_)

# ====================================================
# 7. 聚类热图
# ====================================================

sns.clustermap(
    pearson_corr,
    annot=True,
    cmap='coolwarm',
    figsize=(8,8)
)

plt.savefig(
    r"I:\RP_multib\重新修改\prediction_cluster_heatmap.png",
    dpi=300
)

plt.show()

print("\n✅ 分析完成")