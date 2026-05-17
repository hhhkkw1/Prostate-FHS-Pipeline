rm(list = ls())
options(stringsAsFactors = FALSE, warn = -1)

required_pkgs <- c("jsonlite", "readxl", "dplyr", "pROC", "openxlsx")
for (p in required_pkgs) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

config_path <- file.path("config", "pipeline_config.json")
if (!file.exists(config_path)) stop("Config file not found: config/pipeline_config.json")
cfg <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)

file_path <- cfg$paths$prediction_output_excel
out_eval <- cfg$paths$evaluation_output_excel
plot_dir <- cfg$paths$plot_dir
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

split_col <- cfg$columns$split_col
outcome_col <- cfg$columns$outcome_col
ds_training <- cfg$dataset_values$training
ds_internal <- cfg$dataset_values$internal_val
ds_external <- cfg$dataset_values$external_val

df <- readxl::read_excel(file_path)

required_cols <- c(
  split_col,
  outcome_col,
  "pred_prob_CAPRA",
  "pred_prob_Clinical",
  "pred_prob_ImageModel",
  "pred_prob_Habitat",
  "pred_prob_Combined"
)
if (!all(required_cols %in% names(df))) {
  stop(paste("Missing required columns:", paste(setdiff(required_cols, names(df)), collapse = ", ")))
}

calc_metrics <- function(y_true, y_prob, cutoff = 0.5) {
  y_pred <- ifelse(y_prob >= cutoff, 1, 0)
  cm <- table(factor(y_true, levels = 0:1), factor(y_pred, levels = 0:1))

  tp <- cm[2, 2]; tn <- cm[1, 1]; fp <- cm[1, 2]; fn <- cm[2, 1]
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  ppv <- tp / (tp + fp)
  npv <- tn / (tn + fn)
  acc <- (tp + tn) / sum(cm)

  roc_obj <- pROC::roc(y_true, y_prob, quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  auc_ci <- pROC::ci.auc(roc_obj)

  list(
    sens = sens, spec = spec, ppv = ppv, npv = npv, acc = acc,
    auc = auc_val, auc_lower = auc_ci[1], auc_upper = auc_ci[3], roc_obj = roc_obj
  )
}

calc_continuous_nri <- function(y, p_old, p_new, nboot = 2000, seed = 123) {
  set.seed(seed)
  delta <- p_new - p_old
  event <- y == 1
  nonevent <- y == 0

  nri_event <- mean(delta[event] > 0) - mean(delta[event] < 0)
  nri_nonevent <- mean(delta[nonevent] < 0) - mean(delta[nonevent] > 0)
  nri_obs <- nri_event + nri_nonevent

  n <- length(y)
  boot <- numeric(nboot)
  for (i in seq_len(nboot)) {
    idx <- sample(seq_len(n), n, replace = TRUE)
    d <- p_new[idx] - p_old[idx]
    e <- y[idx] == 1
    ne <- y[idx] == 0
    boot[i] <- (mean(d[e] > 0) - mean(d[e] < 0)) + (mean(d[ne] < 0) - mean(d[ne] > 0))
  }

  se <- sd(boot)
  z <- nri_obs / se
  p <- 2 * (1 - pnorm(abs(z)))
  ci <- quantile(boot, c(0.025, 0.975))
  list(Estimate = nri_obs, CI_low = ci[1], CI_high = ci[2], P_value = p)
}

calc_continuous_idi <- function(y, p_old, p_new, nboot = 2000, seed = 123) {
  set.seed(seed)
  event <- y == 1
  nonevent <- y == 0

  idi_obs <- (mean(p_new[event]) - mean(p_new[nonevent])) - (mean(p_old[event]) - mean(p_old[nonevent]))

  n <- length(y)
  boot <- numeric(nboot)
  for (i in seq_len(nboot)) {
    idx <- sample(seq_len(n), n, replace = TRUE)
    e <- y[idx] == 1
    ne <- y[idx] == 0
    boot[i] <- (mean(p_new[idx][e]) - mean(p_new[idx][ne])) - (mean(p_old[idx][e]) - mean(p_old[idx][ne]))
  }

  se <- sd(boot)
  z <- idi_obs / se
  p <- 2 * (1 - pnorm(abs(z)))
  ci <- quantile(boot, c(0.025, 0.975))
  list(Estimate = idi_obs, CI_low = ci[1], CI_high = ci[2], P_value = p)
}

plot_roc_five_models <- function(data, dataset_name, out_dir, width = 4, height = 4, dpi = 600) {
  y <- data[[outcome_col]]
  keep <- !is.na(y) & y %in% c(0, 1) &
    !is.na(data$pred_prob_CAPRA) &
    !is.na(data$pred_prob_Clinical) &
    !is.na(data$pred_prob_ImageModel) &
    !is.na(data$pred_prob_Habitat) &
    !is.na(data$pred_prob_Combined)
  data <- data[keep, ]
  y <- data[[outcome_col]]
  if (length(unique(y)) < 2) return(invisible(NULL))

  out_file <- file.path(out_dir, paste0("ROC_5models_", dataset_name, ".tiff"))
  tiff(out_file, width = width, height = height, units = "in", res = dpi, compression = "lzw")

  roc_comb <- pROC::roc(y, data$pred_prob_Combined, quiet = TRUE)
  roc_capra <- pROC::roc(y, data$pred_prob_CAPRA, quiet = TRUE)
  roc_clin <- pROC::roc(y, data$pred_prob_Clinical, quiet = TRUE)
  roc_image <- pROC::roc(y, data$pred_prob_ImageModel, quiet = TRUE)
  roc_hab <- pROC::roc(y, data$pred_prob_Habitat, quiet = TRUE)

  plot(roc_comb, col = "black", lwd = 2, legacy.axes = TRUE, xlab = "1 - Specificity", ylab = "Sensitivity",
       main = paste0("ROC Curves (", dataset_name, ")"))
  plot(roc_capra, col = "#1F77B4", lwd = 1.5, add = TRUE)
  plot(roc_clin, col = "#2CA02C", lwd = 1.5, add = TRUE)
  plot(roc_image, col = "#D62728", lwd = 1.5, add = TRUE)
  plot(roc_hab, col = "#FF7F0E", lwd = 1.5, add = TRUE)

  legend(
    "bottomright",
    legend = c(
      paste0("CAPRA (AUC=", round(pROC::auc(roc_capra), 3), ")"),
      paste0("Clinical (AUC=", round(pROC::auc(roc_clin), 3), ")"),
      paste0("ImageModel (AUC=", round(pROC::auc(roc_image), 3), ")"),
      paste0("Habitat (AUC=", round(pROC::auc(roc_hab), 3), ")"),
      paste0("Combined (AUC=", round(pROC::auc(roc_comb), 3), ")")
    ),
    col = c("#1F77B4", "#2CA02C", "#D62728", "#FF7F0E", "black"),
    lwd = c(1.5, 1.5, 1.5, 1.5, 2),
    cex = 0.8,
    bty = "n"
  )
  dev.off()
}

datasets <- c(ds_training, ds_internal, ds_external)
model_cols <- c("pred_prob_CAPRA", "pred_prob_Clinical", "pred_prob_ImageModel", "pred_prob_Habitat", "pred_prob_Combined")
results_all <- list()

for (ds_name in datasets) {
  ds <- df[df[[split_col]] == ds_name, ]
  if (nrow(ds) == 0) next

  y_true <- as.numeric(ds[[outcome_col]])
  keep <- !is.na(y_true) & y_true %in% c(0, 1)
  ds <- ds[keep, ]
  y_true <- as.numeric(ds[[outcome_col]])
  if (length(unique(y_true)) < 2) next

  metrics_list <- lapply(model_cols, function(col) calc_metrics(y_true, ds[[col]]))
  names(metrics_list) <- model_cols

  metrics_df <- do.call(rbind, lapply(names(metrics_list), function(m) {
    x <- metrics_list[[m]]
    data.frame(
      Model = m,
      AUC = x$auc,
      AUC_Lower = x$auc_lower,
      AUC_Upper = x$auc_upper,
      Sensitivity = x$sens,
      Specificity = x$spec,
      PPV = x$ppv,
      NPV = x$npv,
      Accuracy = x$acc
    )
  }))

  comp_models <- c("pred_prob_CAPRA", "pred_prob_Clinical", "pred_prob_ImageModel", "pred_prob_Habitat")
  comp_df <- do.call(rbind, lapply(comp_models, function(comp) {
    roc_comb <- pROC::roc(y_true, ds$pred_prob_Combined, quiet = TRUE)
    roc_other <- pROC::roc(y_true, ds[[comp]], quiet = TRUE)
    delong_p <- pROC::roc.test(roc_comb, roc_other)$p.value

    nri_res <- calc_continuous_nri(y_true, ds[[comp]], ds$pred_prob_Combined, nboot = 2000)
    idi_res <- calc_continuous_idi(y_true, ds[[comp]], ds$pred_prob_Combined, nboot = 2000)

    data.frame(
      Comparison = paste0("Combined vs ", comp),
      Delong_p = delong_p,
      NRI = round(nri_res$Estimate, 4),
      NRI_CI = paste0("(", round(nri_res$CI_low, 4), "-", round(nri_res$CI_high, 4), ")"),
      NRI_p = nri_res$P_value,
      IDI = round(idi_res$Estimate, 4),
      IDI_CI = paste0("(", round(idi_res$CI_low, 4), "-", round(idi_res$CI_high, 4), ")"),
      IDI_p = idi_res$P_value
    )
  }))

  results_all[[ds_name]] <- list(Metrics = metrics_df, Comparison = comp_df)
  plot_roc_five_models(ds, ds_name, plot_dir)
}

wb <- openxlsx::createWorkbook()
for (ds_name in names(results_all)) {
  openxlsx::addWorksheet(wb, paste0(ds_name, "_Metrics"))
  openxlsx::writeData(wb, paste0(ds_name, "_Metrics"), results_all[[ds_name]]$Metrics)
  openxlsx::addWorksheet(wb, paste0(ds_name, "_Comparison"))
  openxlsx::writeData(wb, paste0(ds_name, "_Comparison"), results_all[[ds_name]]$Comparison)
}
openxlsx::saveWorkbook(wb, out_eval, overwrite = TRUE)

cat("Saved:", out_eval, "\n")
cat("ROC plots saved in:", plot_dir, "\n")
