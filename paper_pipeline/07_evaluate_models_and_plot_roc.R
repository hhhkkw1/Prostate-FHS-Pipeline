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
  "pred_prob_Clin_PIRADS",
  "pred_prob_Combined"
)
if (!all(required_cols %in% names(df))) {
  stop(paste("Missing required columns:", paste(setdiff(required_cols, names(df)), collapse = ", ")))
}

get_youden_cutoff <- function(y_true, y_prob) {
  keep <- !is.na(y_true) & !is.na(y_prob) & y_true %in% c(0, 1)
  y <- y_true[keep]
  p <- y_prob[keep]
  if (length(unique(y)) < 2) return(0.5)
  roc_obj <- pROC::roc(y, p, quiet = TRUE)
  coords <- pROC::coords(roc_obj, "best", best.method = "youden")
  return(as.numeric(coords$threshold[1]))
}

calc_metrics <- function(y_true, y_prob, cutoff = NULL) {
  keep <- !is.na(y_true) & !is.na(y_prob) & y_true %in% c(0, 1)
  y <- y_true[keep]
  p <- y_prob[keep]

  if (length(y) == 0) {
    return(list(sens = NA, spec = NA, ppv = NA, npv = NA, acc = NA,
                auc = NA, auc_lower = NA, auc_upper = NA, roc_obj = NULL, cutoff = NA))
  }

  roc_obj <- pROC::roc(y, p, quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  auc_ci <- pROC::ci.auc(roc_obj)

  if (is.null(cutoff)) {
    coords <- pROC::coords(roc_obj, "best", best.method = "youden")
    cutoff <- as.numeric(coords$threshold[1])
  }

  y_pred <- ifelse(p >= cutoff, 1, 0)
  cm <- table(factor(y, levels = 0:1), factor(y_pred, levels = 0:1))
  tp <- cm[2, 2]; tn <- cm[1, 1]; fp <- cm[1, 2]; fn <- cm[2, 1]
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  ppv <- tp / (tp + fp)
  npv <- tn / (tn + fn)
  acc <- (tp + tn) / sum(cm)

  list(sens = sens, spec = spec, ppv = ppv, npv = npv, acc = acc,
       auc = auc_val, auc_lower = auc_ci[1], auc_upper = auc_ci[3],
       roc_obj = roc_obj, cutoff = cutoff)
}

calc_brier <- function(y_true, y_prob) {
  keep <- !is.na(y_true) & !is.na(y_prob) & y_true %in% c(0, 1)
  y <- as.numeric(y_true[keep])
  p <- as.numeric(y_prob[keep])
  if (length(y) == 0) return(NA)
  return(mean((p - y)^2))
}

calc_calibration <- function(y_true, y_prob) {
  keep <- !is.na(y_true) & !is.na(y_prob) & y_true %in% c(0, 1)
  y <- as.numeric(y_true[keep])
  p <- as.numeric(y_prob[keep])
  if (length(y) < 10) return(list(intercept = NA, slope = NA))
  p_clip <- pmax(pmin(p, 1 - 1e-6), 1e-6)
  lp <- log(p_clip / (1 - p_clip))
  tryCatch({
    fit <- glm(y ~ lp, family = binomial)
    return(list(intercept = as.numeric(coef(fit)[1]), slope = as.numeric(coef(fit)[2])))
  }, error = function(e) return(list(intercept = NA, slope = NA)))
}

brier_diff_test <- function(y, p_old, p_new, nboot = 2000, seed = 123) {
  set.seed(seed)
  n <- length(y)
  diff_obs <- calc_brier(y, p_old) - calc_brier(y, p_new)
  diff_boot <- numeric(nboot)
  for (i in seq_len(nboot)) {
    idx <- sample(seq_len(n), n, replace = TRUE)
    diff_boot[i] <- calc_brier(y[idx], p_old[idx]) - calc_brier(y[idx], p_new[idx])
  }
  se <- sd(diff_boot)
  z <- diff_obs / se
  p <- 2 * (1 - pnorm(abs(z)))
  ci <- quantile(diff_boot, c(0.025, 0.975))
  return(list(Diff = diff_obs, CI_low = ci[1], CI_high = ci[2], P_value = p))
}

plot_roc_five_models <- function(data, dataset_name, out_dir, width = 4, height = 4, dpi = 600) {
  y <- data[[outcome_col]]
  keep <- !is.na(y) & y %in% c(0, 1) &
    !is.na(data$pred_prob_CAPRA) &
    !is.na(data$pred_prob_Clinical) &
    !is.na(data$pred_prob_Habitat) &
    !is.na(data$pred_prob_Clin_PIRADS) &
    !is.na(data$pred_prob_Combined)
  data <- data[keep, ]
  y <- data[[outcome_col]]
  if (length(unique(y)) < 2) return(invisible(NULL))

  out_file <- file.path(out_dir, paste0("ROC_5models_", dataset_name, ".tiff"))
  tiff(out_file, width = width, height = height, units = "in", res = dpi, compression = "lzw")

  roc_comb <- pROC::roc(y, data$pred_prob_Combined, quiet = TRUE)
  roc_capra <- pROC::roc(y, data$pred_prob_CAPRA, quiet = TRUE)
  roc_clin <- pROC::roc(y, data$pred_prob_Clinical, quiet = TRUE)
  roc_clin_pirads <- pROC::roc(y, data$pred_prob_Clin_PIRADS, quiet = TRUE)
  roc_hab <- pROC::roc(y, data$pred_prob_Habitat, quiet = TRUE)

  plot(roc_comb, col = "black", lwd = 2, legacy.axes = TRUE, xlab = "1 - Specificity", ylab = "Sensitivity",
       main = paste0("ROC Curves (", dataset_name, ")"))
  plot(roc_capra, col = "#1F77B4", lwd = 1.5, add = TRUE)
  plot(roc_clin, col = "#2CA02C", lwd = 1.5, add = TRUE)
  plot(roc_clin_pirads, col = "#9467BD", lwd = 1.5, add = TRUE)
  plot(roc_hab, col = "#FF7F0E", lwd = 1.5, add = TRUE)

  legend(
    "bottomright",
    legend = c(
      paste0("CAPRA (AUC=", round(pROC::auc(roc_capra), 3), ")"),
      paste0("Clinical (AUC=", round(pROC::auc(roc_clin), 3), ")"),
      paste0("Clin+Img (AUC=", round(pROC::auc(roc_clin_pirads), 3), ")"),
      paste0("FHS (AUC=", round(pROC::auc(roc_hab), 3), ")"),
      paste0("FHS-integrated (AUC=", round(pROC::auc(roc_comb), 3), ")")
    ),
    col = c("#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "black"),
    lwd = c(1.5, 1.5, 1.5, 1.5, 2),
    cex = 0.8,
    bty = "n"
  )
  dev.off()
}

datasets <- c(ds_training, ds_internal, ds_external)
model_cols <- c("pred_prob_CAPRA", "pred_prob_Clinical", "pred_prob_ImageModel",
                "pred_prob_Habitat", "pred_prob_Clin_PIRADS", "pred_prob_Combined")

# Youden cutoff from training set
train_ds <- df[df[[split_col]] == ds_training, ]
train_y <- as.numeric(train_ds[[outcome_col]])
youden_cutoffs <- sapply(model_cols, function(col) get_youden_cutoff(train_y, train_ds[[col]]))
cat("\nTraining set Youden cutoffs:\n")
print(round(youden_cutoffs, 4))

results_all <- list()

for (ds_name in datasets) {
  ds <- df[df[[split_col]] == ds_name, ]
  if (nrow(ds) == 0) next

  y_true <- as.numeric(ds[[outcome_col]])
  keep <- !is.na(y_true) & y_true %in% c(0, 1)
  ds <- ds[keep, ]
  y_true <- as.numeric(ds[[outcome_col]])
  if (length(unique(y_true)) < 2) next

  metrics_list <- lapply(model_cols, function(col) {
    if (ds_name == ds_training) {
      calc_metrics(y_true, ds[[col]])
    } else {
      calc_metrics(y_true, ds[[col]], cutoff = youden_cutoffs[col])
    }
  })
  names(metrics_list) <- model_cols

  metrics_df <- do.call(rbind, lapply(names(metrics_list), function(m) {
    x <- metrics_list[[m]]
    cal <- calc_calibration(y_true, ds[[m]])
    brier <- calc_brier(y_true, ds[[m]])
    data.frame(
      Model = m,
      Youden_Cutoff = x$cutoff,
      AUC = x$auc,
      AUC_Lower = x$auc_lower,
      AUC_Upper = x$auc_upper,
      Brier = round(brier, 4),
      Cal_Intercept = round(cal$intercept, 4),
      Cal_Slope = round(cal$slope, 4),
      Sensitivity = x$sens,
      Specificity = x$spec,
      PPV = x$ppv,
      NPV = x$npv,
      Accuracy = x$acc
    )
  }))

  comp_models <- c("pred_prob_CAPRA", "pred_prob_Clinical", "pred_prob_ImageModel",
                    "pred_prob_Habitat", "pred_prob_Clin_PIRADS")
  comp_df <- do.call(rbind, lapply(comp_models, function(comp) {
    roc_comb <- pROC::roc(y_true, ds$pred_prob_Combined, quiet = TRUE)
    roc_other <- pROC::roc(y_true, ds[[comp]], quiet = TRUE)
    delong_p <- pROC::roc.test(roc_comb, roc_other)$p.value

    brier_res <- brier_diff_test(y_true, ds[[comp]], ds$pred_prob_Combined, nboot = 2000)
    cal_new <- calc_calibration(y_true, ds$pred_prob_Combined)
    cal_old <- calc_calibration(y_true, ds[[comp]])

    data.frame(
      Comparison = paste0("Combined vs ", comp),
      Delong_p = round(delong_p, 4),
      Brier_Diff = round(brier_res$Diff, 4),
      Brier_Diff_CI = paste0("(", round(brier_res$CI_low, 4), "-", round(brier_res$CI_high, 4), ")"),
      Brier_Diff_p = round(brier_res$P_value, 4),
      Cal_Int_Comb = round(cal_new$intercept, 4),
      Cal_Int_Other = round(cal_old$intercept, 4),
      Cal_Slp_Comb = round(cal_new$slope, 4),
      Cal_Slp_Other = round(cal_old$slope, 4)
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
