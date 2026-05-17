# ==============================================================================
# 0. 环境设置与加载包
# ==============================================================================
rm(list = ls()) # 清空环境
options(warn = -1)

# 检查并安装 ggplot2 (如果尚未安装)
if(!require(ggplot2)) install.packages("ggplot2")

library(readxl)
library(openxlsx)
library(dplyr)
library(glmnet)
library(pROC)
library(caret)
library(ggplot2) # 新增绘图包

# ==============================================================================
# 1. 配置与数据加载
# ==============================================================================
FILE_PATH <- "I:/RP_multib/重新修改/副本ALL911改11.xlsx"
OUTPUT_PATH <- gsub(".xlsx", "_Final_LASSO_Result_test.xlsx", FILE_PATH)

cat("📂 [Step 1] 正在读取数据...\n")
df <- read_excel(FILE_PATH)

sub_outcomes <- c('EPE', 'SVI', 'Margin', 'High_GG') 
rad_pattern <- "^h[0-3]_"

# ==============================================================================
# 2. 数据清洗与标签构建
# ==============================================================================
cat("🛠️ [Step 2] 数据准备...\n")

# A. 标签构建
df$Lymph_Calc <- ifelse(is.na(df$Lymph), 0, df$Lymph)
ap_components <- df[, c(sub_outcomes, "Lymph_Calc")]
df$AP_Status <- as.integer(rowSums(ap_components, na.rm = TRUE) > 0)
cat(sprintf("   -> 总体 AP 阳性率: %.2f%%\n", mean(df$AP_Status)*100))

# B. T分期处理
t_map <- c('T1'=0, 'T1a'=0, 'T1b'=0, 'T1c'=0, 'T2a'=1, 'T2b'=2, 'T2c'=3, 'T3a'=4, 'T3b'=5, 'T4'=6)
if(is.character(df$T_new)) {
  df$T_score <- as.numeric(t_map[trimws(df$T_new)])
} else {
  df$T_score <- df$T_new
}

# C. 提取生境特征
rad_features <- grep(rad_pattern, names(df), value = TRUE)

# D. 缺失值填充 (Training Median)
train_idx <- which(df$Dataset_Type == 'Training')
vars_to_fill <- c(rad_features, "T_score")
for(col in vars_to_fill) {
  med_val <- median(df[[col]][train_idx], na.rm = TRUE)
  df[[col]][is.na(df[[col]]) | is.infinite(df[[col]])] <- med_val
}

# ==============================================================================
# 3. 多重特征筛选 (Training Set Only)
# ==============================================================================
cat("\n🔍 [Step 3] 开始漏斗式特征筛选 (仅基于训练集)...\n")

X_train_raw <- as.matrix(df[train_idx, rad_features])
y_train <- df$AP_Status[train_idx]
current_feats <- rad_features

# --- A. 方差筛选 ---
nzv <- nearZeroVar(X_train_raw)
if(length(nzv) > 0) {
  dropped_nzv <- colnames(X_train_raw)[nzv]
  current_feats <- setdiff(current_feats, dropped_nzv)
  cat(sprintf("   [1/3] 方差筛选: 剔除 %d 个 -> 剩余 %d 个\n", length(nzv), length(current_feats)))
} else {
  cat(sprintf("   [1/3] 方差筛选: 无剔除 -> 剩余 %d 个\n", length(current_feats)))
}

X_curr <- X_train_raw[, current_feats]

# --- B. 相关性去重 (AUC指导) ---
calc_auc <- function(x, y) { max(as.numeric(auc(roc(y, x, quiet=TRUE))), as.numeric(auc(roc(y, -x, quiet=TRUE)))) }
auc_scores <- apply(X_curr, 2, calc_auc, y=y_train)

corr_mat <- cor(X_curr, method="spearman")
drop_cols <- c()
cols <- colnames(X_curr)

for(i in 1:(length(cols)-1)) {
  for(j in (i+1):length(cols)) {
    c1 <- cols[i]; c2 <- cols[j]
    if(c1 %in% drop_cols || c2 %in% drop_cols) next
    # 按照您提供的代码逻辑，这里使用 0.9 作为阈值
    if(abs(corr_mat[i,j]) > 0.9) { 
      if(auc_scores[c1] >= auc_scores[c2]) drop_cols <- c(drop_cols, c2)
      else drop_cols <- c(drop_cols, c1)
    }
  }
}

current_feats <- setdiff(current_feats, drop_cols)
cat(sprintf("   [2/3] 共线性筛选(>0.9): 剔除 %d 个 -> 剩余 %d 个\n", length(drop_cols), length(current_feats)))
X_curr <- X_curr[, current_feats]

# --- C. 单因素 MWU 检验 ---
cat("   [3/3] 单因素 MWU 检验 (P < 0.05)...\n")
p_values <- c()
for(feat in current_feats) {
  wt <- wilcox.test(X_curr[, feat] ~ y_train)
  p_values[feat] <- wt$p.value
}

significant_feats <- names(p_values)[p_values < 0.05]
dropped_p <- length(current_feats) - length(significant_feats)

current_feats <- significant_feats
cat(sprintf("         剔除 P>=0.05 特征: %d 个 -> 最终入模 %d 个\n", dropped_p, length(current_feats)))

X_train_final <- X_curr[, current_feats]

# ==============================================================================
# 4. LASSO 建模 (Deviance Mode)
# ==============================================================================
cat("\n🔮 [Step 4] 运行 LASSO (Binomial Deviance)...\n")

preProc <- preProcess(as.data.frame(X_train_final), method = c("center", "scale"))
X_train_scaled <- predict(preProc, as.data.frame(X_train_final))
X_train_mat <- as.matrix(X_train_scaled)

set.seed(42)
cv_fit <- cv.glmnet(
  x = X_train_mat, 
  y = y_train, 
  family = "binomial", 
  type.measure = "deviance",
  nfolds = 10,
  alpha = 1 
)

cat(sprintf("   -> Min Deviance: %.4f\n", min(cv_fit$cvm)))

# ==============================================================================
# 5. 提取结果 & 准备保存系数 (重要修改)
# ==============================================================================
# 获取 1se 标准下的所有系数(包含截距)
coef_obj <- coef(cv_fit, s = "lambda.1se")
# 转换为数据框
coef_df_all <- data.frame(
  Feature = rownames(coef_obj), 
  Coef = as.matrix(coef_obj)[,1]
)

# 提取截距
intercept_val <- coef_df_all$Coef[coef_df_all$Feature == "(Intercept)"]

# 提取非零特征(用于展示和绘图)
feats_1se <- coef_df_all %>% 
  filter(Coef != 0, Feature != "(Intercept)") %>% 
  arrange(desc(abs(Coef)))

cat("\n🏆 [推荐] 1se 标准选中的特征:\n")
print(feats_1se)
cat(sprintf("   -> Intercept (截距): %.4f\n", intercept_val))

# ==============================================================================
# 6. 计算得分、绘图与保存 (重要修改)
# ==============================================================================
cat("\n💾 [Step 6] 结果保存与可视化...\n")

# A. 计算得分
X_all_raw <- df[, current_feats]
X_all_scaled <- predict(preProc, X_all_raw)
X_all_mat <- as.matrix(X_all_scaled)

df$Rad_score_1se <- as.numeric(predict(cv_fit, newx = X_all_mat, s = "lambda.1se"))

# 单独输出文件路径（12:8，TIFF）
PLOT_CV_PATH <- gsub(".xlsx", "_Plot1_CV_12x8.tiff", FILE_PATH)
PLOT_PATH_PATH <- gsub(".xlsx", "_Plot2_Path_12x8.tiff", FILE_PATH)
PLOT_IMPORTANCE_PATH <- gsub(".xlsx", "_Plot3_Importance_12x8.tiff", FILE_PATH)

# 统一字体加粗 + 放大参数
base_cex <- 1.7
main_cex <- 2.1
lab_cex <- 1.8
axis_cex <- 1.7

# 统一线宽参数（进一步加粗）
axis_lwd <- 2.2
curve_lwd <- 2.8
ref_lwd <- 2.2
grid_lwd <- 1.8

# B. 保存 PDF (新增特征重要性图)
pdf(gsub(".xlsx", "_Plots_Final.pdf", FILE_PATH), width = 14, height = 8)

# 布局: 左边两个正方形图，右边一个长条图
layout(matrix(c(1, 2, 3, 3), nrow = 2, byrow = FALSE)) 
par(
  cex = base_cex, cex.main = main_cex, cex.lab = lab_cex, cex.axis = axis_cex,
  font = 2, font.main = 2, font.lab = 2, font.axis = 2
)
par(lwd = axis_lwd)

# Plot 1: CV Plot
plot(cv_fit, lwd = curve_lwd)
title("LASSO CV", line = 2.5)

# Plot 2: Path Plot
plot(cv_fit$glmnet.fit, xvar = "lambda", label = TRUE, lwd = curve_lwd)
title("Coefficient Path", line = 2.5)
abline(v = log(cv_fit$lambda.1se), col = "red", lwd = ref_lwd, lty = 2)
abline(v = log(cv_fit$lambda.min), col = "blue", lwd = ref_lwd, lty = 2)
# Plot 3: Feature Importance Barplot (使用 ggplot2 风格)
# 为了在 layout 中使用 ggplot，我们需要恢复默认设置或使用 grid
# 这里使用 base R 模拟漂亮的柱状图，以兼容 layout
par(
  mar = c(5, 16, 4, 2), # 进一步增加左边距，避免特征名被截断
  mgp = c(3, 1.2, 0),
  xpd = NA,
  cex = base_cex, cex.main = main_cex, cex.lab = lab_cex, cex.axis = axis_cex,
  font = 2, font.main = 2, font.lab = 2, font.axis = 2
)
par(lwd = axis_lwd)
cols <- ifelse(feats_1se$Coef > 0, "#E64B35", "#4DBBD5") # 红/蓝配色
barplot(
  rev(feats_1se$Coef), # 反转顺序，大的在上面
  names.arg = rev(feats_1se$Feature),
  horiz = TRUE,
  las = 1, # 文字水平显示
  col = rev(cols),
  border = "black",
  lwd = 1.2,
  main = "Feature Importance",
  xlab = "Coefficient Value",
  xlim = c(min(feats_1se$Coef)*1.2, max(feats_1se$Coef)*1.2) # 调整X轴范围
)
grid(ny = NA, nx = NULL, col = "gray35", lty = "dotted", lwd = grid_lwd)

dev.off()

# B-1. 单独导出 Plot 1: CV Plot (12:8, TIFF, 字体加粗增大)
tiff(PLOT_CV_PATH, width = 12, height = 8, units = "in", res = 300, compression = "lzw")
par(
  cex = base_cex, cex.main = main_cex, cex.lab = lab_cex, cex.axis = axis_cex,
  font = 2, font.main = 2, font.lab = 2, font.axis = 2
)
par(lwd = axis_lwd)
plot(cv_fit, lwd = curve_lwd)
title("LASSO CV", line = 2.5)
dev.off()

# B-2. 单独导出 Plot 2: Path Plot (12:8, TIFF, 字体加粗增大)
tiff(PLOT_PATH_PATH, width = 12, height = 8, units = "in", res = 300, compression = "lzw")
par(
  cex = base_cex, cex.main = main_cex, cex.lab = lab_cex, cex.axis = axis_cex,
  font = 2, font.main = 2, font.lab = 2, font.axis = 2
)
par(lwd = axis_lwd)
plot(cv_fit$glmnet.fit, xvar = "lambda", label = TRUE, lwd = curve_lwd)
title("Coefficient Path", line = 2.5)
abline(v = log(cv_fit$lambda.1se), col = "red", lwd = ref_lwd, lty = 2)
abline(v = log(cv_fit$lambda.min), col = "blue", lwd = ref_lwd, lty = 2)
dev.off()

# B-3. 单独导出 Plot 3: Feature Importance (12:8, TIFF, 字体加粗增大)
tiff(PLOT_IMPORTANCE_PATH, width = 14, height = 8, units = "in", res = 300, compression = "lzw")
par(
  mar = c(5, 16, 4, 2), mgp = c(3, 1.2, 0), xpd = NA,
  cex = base_cex, cex.main = main_cex, cex.lab = lab_cex, cex.axis = axis_cex,
  font = 2, font.main = 2, font.lab = 2, font.axis = 2
)
par(lwd = axis_lwd)
barplot(
  rev(feats_1se$Coef),
  names.arg = rev(feats_1se$Feature),
  horiz = TRUE,
  las = 1,
  col = rev(cols),
  border = "black",
  lwd = 1.2,
  main = "Feature Importance",
  xlab = "Coefficient Value",
  xlim = c(min(feats_1se$Coef)*1.2, max(feats_1se$Coef)*1.2)
)
grid(ny = NA, nx = NULL, col = "gray35", lty = "dotted", lwd = grid_lwd)
dev.off()

# C. 保存 Excel (多Sheet模式)
wb <- createWorkbook()

# Sheet 1: 完整数据
addWorksheet(wb, "Data")
writeData(wb, "Data", df)

# Sheet 2: 模型系数
addWorksheet(wb, "Model_Coefs")
# 构建一个漂亮的系数表
final_coef_table <- rbind(
  data.frame(Feature = "(Intercept)", Coefficient = intercept_val),
  feats_1se %>% rename(Coefficient = Coef)
)
writeData(wb, "Model_Coefs", final_coef_table)

saveWorkbook(wb, OUTPUT_PATH, overwrite = TRUE)

cat(sprintf("✅ 全部完成！\n"))
cat(sprintf("   -> 数据文件: %s (含 Data 和 Model_Coefs 两个Sheet)\n", OUTPUT_PATH))
cat(sprintf("   -> 图表文件: %s (含特征重要性图)\n", gsub(".xlsx", "_Plots_Final_test.pdf", FILE_PATH)))
cat(sprintf("   -> 单图文件: %s\n", PLOT_CV_PATH))
cat(sprintf("   -> 单图文件: %s\n", PLOT_PATH_PATH))
cat(sprintf("   -> 单图文件: %s\n", PLOT_IMPORTANCE_PATH))


