# ==============================================================================
# 四模型 ROC 曲线（TIFF，无右/上边框，尺寸可控）
# ==============================================================================

rm(list = ls())
options(stringsAsFactors = FALSE)

library(readxl)
library(dplyr)
library(pROC)

# -------------------------------
# 1. 读取数据
# -------------------------------
df <- read_excel("I:/RP_multib/副本ALL911_Predictions.xlsx")

# -------------------------------
# 2. 绘图函数
# -------------------------------
plot_roc_four_models <- function(data, dataset_name,
                                 width = 6, height = 6, dpi = 600){
  
  y <- data$AP_Status
  
  # 打开 TIFF 设备
  tiff(
    filename = paste0("I:/RP_multib/ROC_4models_", dataset_name, ".tiff"),
    width = width,
    height = height,
    units = "in",
    res = dpi,
    compression = "lzw"
  )
  
  par(bty = "n")   # 关键：彻底去掉所有边框
  
  # ✅ 全局字体控制
  par(
    bty = "n",
    cex.axis = 1,
    cex.lab  = 1,
    cex.main = 1,
    lwd=4,
    lwd.ticks=0.8,
    tck=-0.01
  )
  
  # Combined 先画
  roc_comb <- roc(y, data$pred_prob_Combined)
  plot(
    roc_comb,
    col = "black",
    lwd = 2,
    legacy.axes = TRUE,
    xlab = "1 - Specificity",
    ylab = "Sensitivity",
    main = paste0("ROC Curves (", dataset_name, ")")
  )
  
  # 其他模型
  roc_capra   <- roc(y, data$pred_prob_CAPRA)
  roc_clin    <- roc(y, data$pred_prob_Clinical)
  roc_habitat <- roc(y, data$pred_prob_Habitat)
  
  plot(roc_capra,   col = "#1F77B4", lwd = 1.5, add = TRUE)
  plot(roc_clin,    col = "#2CA02C", lwd = 1.5, add = TRUE)
  plot(roc_habitat, col = "#FF7F0E", lwd = 1.5, add = TRUE)
  
  # 手动画左 & 下边框
  #box(bty = "l")
  
  legend(
    "bottomright",
    legend = c(
      paste0("CAPRA (AUC=", round(auc(roc_capra),3),")"),
      paste0("Clinical (AUC=", round(auc(roc_clin),3),")"),
      paste0("Habitat (AUC=", round(auc(roc_habitat),3),")"),
      paste0("Combined (AUC=", round(auc(roc_comb),3),")")
    ),
    col = c("#1F77B4", "#2CA02C", "#FF7F0E", "black"),
    lwd = c(1.5,1.5,1.5,2),
    cex = 0.8,
    bty = "n"
  )
  
  dev.off()
}

# -------------------------------
# 3. Internal / External 分别绘制
# -------------------------------
for(ds in c("Internal_Val", "External_Val")){
  df_ds <- df %>% filter(Dataset_Type == ds)
  if(nrow(df_ds) == 0) next
  
  plot_roc_four_models(
    data = df_ds,
    dataset_name = ds,
    width = 4,    # ← 你可以自由改
    height = 4,   # ← 你可以自由改
    dpi = 600
  )
}

cat("✅ ROC TIFF 图像已生成（无右/上边框，尺寸可控）\n")
