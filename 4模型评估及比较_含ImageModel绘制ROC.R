# ==============================================================================
# 模型评估脚本（兼容 pROC 1.18.5，去掉雷达图，增加ROC曲线）
# ==============================================================================
rm(list=ls())
options(stringsAsFactors = FALSE, warn=-1)

# -------------------------------
# 0. 载入包
# -------------------------------
packages <- c("readxl","dplyr","pROC","rmda","ggplot2","openxlsx","DescTools","nricens","scales","rms","PredictABEL")
for(p in packages){
  if(!require(p, character.only = TRUE)){
    install.packages(p, dependencies=TRUE)
    library(p, character.only = TRUE)
  }
}

# -------------------------------
# 1. 读取数据
# -------------------------------
file_path <- "I:/RP_multib/重新修改/副本ALL911改11.xlsx"
df <- readxl::read_excel(file_path)

# 兼容旧列名：pred_prob_Image -> pred_prob_ImageModel
if(!("pred_prob_ImageModel" %in% names(df)) && ("pred_prob_Image" %in% names(df))){
  df$pred_prob_ImageModel <- df$pred_prob_Image
}

required_cols <- c("Dataset_Type","AP_Status","pred_prob_Clinical","pred_prob_ImageModel","pred_prob_Habitat",
                   "pred_prob_Combined","pred_prob_CAPRA")
if(!all(required_cols %in% names(df))){
  stop("❌ 数据缺少必要列，请检查")
}

# -------------------------------
# 2. 指标计算函数
# -------------------------------
calc_metrics <- function(y_true, y_prob, cutoff=0.5){
  y_true <- as.numeric(y_true)
  y_prob <- as.numeric(y_prob)
  
  if(length(y_true) != length(y_prob)){
    stop(sprintf("Length mismatch in calc_metrics: y_true=%d, y_prob=%d", length(y_true), length(y_prob)))
  }
  
  keep <- !is.na(y_true) & !is.na(y_prob) & y_true %in% c(0,1)
  y_true <- y_true[keep]
  y_prob <- y_prob[keep]
  
  if(length(y_true) == 0){
    return(list(sens=NA, spec=NA, ppv=NA, npv=NA, acc=NA,
                auc=NA, auc_lower=NA, auc_upper=NA, roc_obj=NULL))
  }
  
  y_pred <- ifelse(y_prob >= cutoff, 1, 0)
  cm <- table(factor(y_true,levels=0:1), factor(y_pred, levels=0:1))
  TP <- cm[2,2]; TN <- cm[1,1]; FP <- cm[1,2]; FN <- cm[2,1]
  sens <- TP / (TP+FN)
  spec <- TN / (TN+FP)
  ppv  <- TP / (TP+FP)
  npv  <- TN / (TN+FN)
  acc  <- (TP+TN)/sum(cm)
  
  # ROC（兼容旧版 pROC 1.18.5，不使用 quiet 参数）
  roc_obj <- pROC::roc(y_true, y_prob)
  auc_val <- pROC::auc(roc_obj)
  auc_ci <- pROC::ci.auc(roc_obj)
  
  return(list(sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc,
              auc=as.numeric(auc_val), auc_lower=auc_ci[1], auc_upper=auc_ci[3],
              roc_obj=roc_obj))
}

calc_continuous_NRI <- function(y, p_old, p_new, nboot = 1000, seed = 123){
  
  set.seed(seed)
  
  delta <- p_new - p_old
  event <- y == 1
  nonevent <- y == 0
  
  # observed NRI
  nri_event <- mean(delta[event] > 0) - mean(delta[event] < 0)
  nri_nonevent <- mean(delta[nonevent] < 0) - mean(delta[nonevent] > 0)
  nri_obs <- nri_event + nri_nonevent
  
  # bootstrap
  n <- length(y)
  nri_boot <- numeric(nboot)
  
  for(i in seq_len(nboot)){
    idx <- sample(seq_len(n), n, replace = TRUE)
    d <- p_new[idx] - p_old[idx]
    e <- y[idx] == 1
    ne <- y[idx] == 0
    
    nri_e <- mean(d[e] > 0) - mean(d[e] < 0)
    nri_ne <- mean(d[ne] < 0) - mean(d[ne] > 0)
    nri_boot[i] <- nri_e + nri_ne
  }
  
  se <- sd(nri_boot)
  z  <- nri_obs / se
  p  <- 2 * (1 - pnorm(abs(z)))
  ci <- quantile(nri_boot, c(0.025, 0.975))
  
  return(list(
    Estimate = nri_obs,
    CI_low   = ci[1],
    CI_high  = ci[2],
    P_value  = p
  ))
}

calc_continuous_IDI <- function(y, p_old, p_new, nboot = 1000, seed = 123){
  
  set.seed(seed)
  
  event <- y == 1
  nonevent <- y == 0
  
  idi_obs <- (mean(p_new[event]) - mean(p_new[nonevent])) -
    (mean(p_old[event]) - mean(p_old[nonevent]))
  
  n <- length(y)
  idi_boot <- numeric(nboot)
  
  for(i in seq_len(nboot)){
    idx <- sample(seq_len(n), n, replace = TRUE)
    e <- y[idx] == 1
    ne <- y[idx] == 0
    
    idi_boot[i] <- (mean(p_new[idx][e]) - mean(p_new[idx][ne])) -
      (mean(p_old[idx][e]) - mean(p_old[idx][ne]))
  }
  
  se <- sd(idi_boot)
  z  <- idi_obs / se
  p  <- 2 * (1 - pnorm(abs(z)))
  ci <- quantile(idi_boot, c(0.025, 0.975))
  
  return(list(
    Estimate = idi_obs,
    CI_low   = ci[1],
    CI_high  = ci[2],
    P_value  = p
  ))
}

# -------------------------------
# 3. 模型比较函数（Combined vs others）
# -------------------------------
compare_models <- function(y_true, p_comb, p_other){
  roc_comb <- pROC::roc(y_true, p_comb)
  roc_other <- pROC::roc(y_true, p_other)
  delong_p <- tryCatch({pROC::roc.test(roc_comb, roc_other)$p.value}, error=function(e) NA)
  
  nri_idi <- tryCatch({
    nricens::NRI(y_true, p_other, p_comb, updown="categoryless", niter=1000)
  }, error=function(e) NA)
  
  return(list(Delong_p=delong_p, NRI=nri_idi))
}

# -------------------------------
# 4. 循环评估
# -------------------------------
datasets <- c("Training","Internal_Val","External_Val")
model_cols <- c("pred_prob_CAPRA","pred_prob_Clinical","pred_prob_ImageModel","pred_prob_Habitat","pred_prob_Combined")
results_all <- list()

for(ds_name in datasets){
  ds <- df[df$Dataset_Type==ds_name, ]
  if(nrow(ds)==0) next
  y_true <- as.numeric(ds$AP_Status)
  cat("\n==============================\nDataset:", ds_name, "\n==============================\n")
  
  # 4.1 计算指标
  metrics_list <- lapply(model_cols, function(col) calc_metrics(y_true, ds[[col]]))
  names(metrics_list) <- model_cols
  
  # 保存指标到结构化表格
  metrics_df <- do.call(rbind, lapply(names(metrics_list), function(m){
    x <- metrics_list[[m]]
    data.frame(Model=m, AUC=x$auc, AUC_Lower=x$auc_lower, AUC_Upper=x$auc_upper,
               Sensitivity=x$sens, Specificity=x$spec, PPV=x$ppv, NPV=x$npv, Accuracy=x$acc)
  }))
  
  
  # 4.5 模型比较（Combined vs others）
  # 4.5 模型比较（Combined vs others，continuous NRI / IDI）
  comp_models <- c("pred_prob_CAPRA","pred_prob_Clinical","pred_prob_ImageModel","pred_prob_Habitat")
  
  comp_df <- do.call(rbind, lapply(comp_models, function(comp){
    
    # DeLong
    roc_comb <- pROC::roc(y_true, ds$pred_prob_Combined)
    roc_other <- pROC::roc(y_true, ds[[comp]])
    delong_p <- pROC::roc.test(roc_comb, roc_other)$p.value
    
    # continuous NRI
    nri_res <- calc_continuous_NRI(
      y = y_true,
      p_old = ds[[comp]],
      p_new = ds$pred_prob_Combined,
      nboot = 2000
    )
    
    # continuous IDI
    idi_res <- calc_continuous_IDI(
      y = y_true,
      p_old = ds[[comp]],
      p_new = ds$pred_prob_Combined,
      nboot = 2000
    )
    
    data.frame(
      Comparison = paste0("Combined vs ", comp),
      Delong_p   = delong_p,
      
      NRI        = round(nri_res$Estimate, 4),
      NRI_CI     = paste0("(", round(nri_res$CI_low,4), "-", round(nri_res$CI_high,4), ")"),
      NRI_p      = nri_res$P_value,
      
      IDI        = round(idi_res$Estimate, 4),
      IDI_CI     = paste0("(", round(idi_res$CI_low,4), "-", round(idi_res$CI_high,4), ")"),
      IDI_p      = idi_res$P_value
    )
  }))
  
  
  # 保存所有结果
  results_all[[ds_name]] <- list(Metrics=metrics_df, Comparison=comp_df)
  
  # 输出到控制台
  print(metrics_df)
  print(comp_df)
}

# -------------------------------
# 5. 保存 Excel
# -------------------------------
wb <- openxlsx::createWorkbook()
for(ds_name in names(results_all)){
  openxlsx::addWorksheet(wb, paste0(ds_name,"_Metrics"))
  openxlsx::writeData(wb, sheet=paste0(ds_name,"_Metrics"), results_all[[ds_name]]$Metrics)
  openxlsx::addWorksheet(wb, paste0(ds_name,"_Comparison"))
  openxlsx::writeData(wb, sheet=paste0(ds_name,"_Comparison"), results_all[[ds_name]]$Comparison)
}
openxlsx::saveWorkbook(wb, paste0("I:/RP_multib/重新修改/Model_Evaluation_Results10-1.xlsx"), overwrite=TRUE)

cat("\n✅ 模型评估完成，结果与图表已保存\n")

# -------------------------------
# 6. 绘制ROC曲线（新增 ImageModel）
# -------------------------------
plot_roc_five_models <- function(data, dataset_name,
                                 width = 4, height = 4, dpi = 600){
  y <- data$AP_Status

  # 仅保留结局有效且5个模型预测值非缺失的数据
  keep <- !is.na(y) &
    y %in% c(0, 1) &
    !is.na(data$pred_prob_CAPRA) &
    !is.na(data$pred_prob_Clinical) &
    !is.na(data$pred_prob_ImageModel) &
    !is.na(data$pred_prob_Habitat) &
    !is.na(data$pred_prob_Combined)
  data <- data[keep, ]
  y <- data$AP_Status

  if(length(unique(y)) < 2){
    cat("⚠️", dataset_name, "结局仅单一类别，跳过ROC绘图\n")
    return(invisible(NULL))
  }

  out_file <- paste0("I:/RP_multib/重新修改/ROC_5models_", dataset_name, ".tiff")
  tiff(
    filename = out_file,
    width = width,
    height = height,
    units = "in",
    res = dpi,
    compression = "lzw"
  )

  par(
    bty = "n",
    cex.axis = 1,
    cex.lab = 1,
    cex.main = 1,
    lwd = 4,
    lwd.ticks = 0.8,
    tck = -0.01
  )

  roc_comb <- pROC::roc(y, data$pred_prob_Combined)
  plot(
    roc_comb,
    col = "black",
    lwd = 2,
    legacy.axes = TRUE,
    xlab = "1 - Specificity",
    ylab = "Sensitivity",
    main = paste0("ROC Curves (", dataset_name, ")")
  )

  roc_capra <- pROC::roc(y, data$pred_prob_CAPRA)
  roc_clin <- pROC::roc(y, data$pred_prob_Clinical)
  roc_image <- pROC::roc(y, data$pred_prob_ImageModel)
  roc_habitat <- pROC::roc(y, data$pred_prob_Habitat)

  plot(roc_capra, col = "#1F77B4", lwd = 1.5, add = TRUE)
  plot(roc_clin, col = "#2CA02C", lwd = 1.5, add = TRUE)
  plot(roc_image, col = "#D62728", lwd = 1.5, add = TRUE)
  plot(roc_habitat, col = "#FF7F0E", lwd = 1.5, add = TRUE)

  legend(
    "bottomright",
    legend = c(
      paste0("CAPRA (AUC=", round(pROC::auc(roc_capra), 3), ")"),
      paste0("Clinical (AUC=", round(pROC::auc(roc_clin), 3), ")"),
      paste0("Image (AUC=", round(pROC::auc(roc_image), 3), ")"),
      paste0("Habitat (AUC=", round(pROC::auc(roc_habitat), 3), ")"),
      paste0("Combined (AUC=", round(pROC::auc(roc_comb), 3), ")")
    ),
    col = c("#1F77B4", "#2CA02C", "#D62728", "#FF7F0E", "black"),
    lwd = c(1.5, 1.5, 1.5, 1.5, 2),
    cex = 0.8,
    bty = "n"
  )

  dev.off()
  cat("✅ 已保存ROC图:", out_file, "\n")
}

# 绘制 Training / Internal_Val / External_Val
for(ds in c("Training", "Internal_Val", "External_Val")){
  df_ds <- df %>% dplyr::filter(Dataset_Type == ds)
  if(nrow(df_ds) == 0) next
  plot_roc_five_models(df_ds, ds, width = 4, height = 4, dpi = 600)
}

cat("✅ ROC TIFF 图像已生成（含 ImageModel）\n")
