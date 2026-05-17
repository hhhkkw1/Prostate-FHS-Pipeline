rm(list = ls())
options(stringsAsFactors = FALSE, warn = -1)

required_pkgs <- c("jsonlite", "readxl", "openxlsx", "dplyr", "glmnet", "pROC", "caret")
for (p in required_pkgs) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

config_path <- file.path("config", "pipeline_config.json")
if (!file.exists(config_path)) stop("Config file not found: config/pipeline_config.json")
cfg <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)

infile <- cfg$paths$feature_output_excel
outfile <- cfg$paths$lasso_output_excel
plot_dir <- cfg$paths$plot_dir
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

split_col <- cfg$columns$split_col
outcome_col <- cfg$columns$outcome_col
training_value <- cfg$dataset_values$training
pattern <- cfg$features$habitat_prefix_pattern
outcome_components <- unlist(cfg$outcome_components)
lymph_col <- cfg$lymph_col

df <- readxl::read_excel(infile)

case_id_candidates <- unlist(cfg$columns$case_id_candidates)
case_id_col <- case_id_candidates[case_id_candidates %in% names(df)][1]
if (is.na(case_id_col)) stop("No case ID column detected in feature table.")
if (!("CaseID" %in% names(df))) df$CaseID <- as.character(df[[case_id_col]])
df$CaseID <- trimws(as.character(df$CaseID))

if (!(outcome_col %in% names(df))) {
  missing_parts <- setdiff(c(outcome_components, lymph_col), names(df))
  if (length(missing_parts) > 0) stop(paste("Missing columns to build AP_Status:", paste(missing_parts, collapse = ", ")))
  df$Lymph_Calc <- ifelse(is.na(df[[lymph_col]]), 0, df[[lymph_col]])
  ap_mat <- df[, c(outcome_components, "Lymph_Calc")]
  df[[outcome_col]] <- as.integer(rowSums(ap_mat, na.rm = TRUE) > 0)
}

if (!("T_score" %in% names(df)) && ("T_new" %in% names(df))) {
  t_map <- c("T1" = 0, "T1a" = 0, "T1b" = 0, "T1c" = 0, "T2a" = 1, "T2b" = 2, "T2c" = 3, "T3a" = 4, "T3b" = 5, "T4" = 6)
  df$T_score <- as.numeric(t_map[trimws(as.character(df$T_new))])
}

if (!(split_col %in% names(df))) stop(paste("Missing split column:", split_col))

habitat_features <- grep(pattern, names(df), value = TRUE)
if (length(habitat_features) == 0) stop("No habitat feature columns matched pattern.")

numeric_cols <- unique(c(habitat_features, "T_score", outcome_col))
for (col in numeric_cols) {
  if (col %in% names(df)) df[[col]] <- as.numeric(df[[col]])
}

train_idx <- which(trimws(as.character(df[[split_col]])) == training_value)
if (length(train_idx) == 0) stop("No training cases found.")

vars_to_fill <- intersect(c(habitat_features, "T_score"), names(df))
for (col in vars_to_fill) {
  med <- median(df[[col]][train_idx], na.rm = TRUE)
  if (!is.finite(med)) med <- 0
  miss <- is.na(df[[col]]) | is.infinite(df[[col]])
  df[[col]][miss] <- med
}

train_df <- df[train_idx, ]
train_df <- train_df[!is.na(train_df[[outcome_col]]) & train_df[[outcome_col]] %in% c(0, 1), ]
if (nrow(train_df) == 0) stop("No valid training rows with binary outcome.")

y_train <- as.integer(train_df[[outcome_col]])
if (length(unique(y_train)) < 2) stop("Training outcome has one class only.")

current_feats <- habitat_features
X_train_raw <- as.matrix(train_df[, current_feats, drop = FALSE])

nzv <- caret::nearZeroVar(X_train_raw)
if (length(nzv) > 0) {
  current_feats <- setdiff(current_feats, colnames(X_train_raw)[nzv])
}

X_curr <- as.matrix(train_df[, current_feats, drop = FALSE])
if (ncol(X_curr) < 1) stop("No features left after variance filter.")

calc_auc <- function(x, y) {
  a1 <- as.numeric(pROC::auc(pROC::roc(y, x, quiet = TRUE)))
  a2 <- as.numeric(pROC::auc(pROC::roc(y, -x, quiet = TRUE)))
  max(a1, a2)
}

auc_scores <- apply(X_curr, 2, calc_auc, y = y_train)
corr_mat <- cor(X_curr, method = "spearman")
drop_cols <- c()
cols <- colnames(X_curr)

if (length(cols) > 1) {
  for (i in 1:(length(cols) - 1)) {
    for (j in (i + 1):length(cols)) {
      c1 <- cols[i]; c2 <- cols[j]
      if (c1 %in% drop_cols || c2 %in% drop_cols) next
      if (abs(corr_mat[i, j]) > 0.9) {
        if (auc_scores[c1] >= auc_scores[c2]) drop_cols <- c(drop_cols, c2) else drop_cols <- c(drop_cols, c1)
      }
    }
  }
}

current_feats <- setdiff(current_feats, drop_cols)
X_curr <- as.matrix(train_df[, current_feats, drop = FALSE])

p_values <- c()
for (feat in current_feats) {
  wt <- suppressWarnings(wilcox.test(X_curr[, feat] ~ y_train))
  p_values[feat] <- wt$p.value
}

selected <- names(p_values)[p_values < 0.05]
if (length(selected) == 0) {
  selected <- current_feats
}

X_train_final <- as.data.frame(train_df[, selected, drop = FALSE])
preproc <- caret::preProcess(X_train_final, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train_final)
X_train_mat <- as.matrix(X_train_scaled)

set.seed(42)
cv_fit <- glmnet::cv.glmnet(
  x = X_train_mat,
  y = y_train,
  family = "binomial",
  type.measure = "deviance",
  nfolds = 10,
  alpha = 1
)

coef_obj <- coef(cv_fit, s = "lambda.1se")
coef_df <- data.frame(
  Feature = rownames(coef_obj),
  Coef = as.matrix(coef_obj)[, 1],
  stringsAsFactors = FALSE
)

intercept_val <- coef_df$Coef[coef_df$Feature == "(Intercept)"]
feats_1se <- coef_df %>% dplyr::filter(Coef != 0, Feature != "(Intercept)") %>% dplyr::arrange(desc(abs(Coef)))

X_all <- as.data.frame(df[, selected, drop = FALSE])
X_all_scaled <- predict(preproc, X_all)
df$FHS <- as.numeric(predict(cv_fit, newx = as.matrix(X_all_scaled), s = "lambda.1se"))

wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "Data")
openxlsx::writeData(wb, "Data", df)
openxlsx::addWorksheet(wb, "Model_Coefs")
coef_out <- rbind(data.frame(Feature = "(Intercept)", Coefficient = intercept_val), feats_1se %>% dplyr::rename(Coefficient = Coef))
openxlsx::writeData(wb, "Model_Coefs", coef_out)
openxlsx::addWorksheet(wb, "Selected_Features")
openxlsx::writeData(wb, "Selected_Features", data.frame(Feature = selected))
openxlsx::saveWorkbook(wb, outfile, overwrite = TRUE)

plot_cv <- file.path(plot_dir, "lasso_cv_curve.tiff")
plot_path <- file.path(plot_dir, "lasso_path_curve.tiff")
plot_imp <- file.path(plot_dir, "lasso_feature_importance.tiff")

tiff(plot_cv, width = 12, height = 8, units = "in", res = 300, compression = "lzw")
plot(cv_fit)
title("LASSO CV")
dev.off()

tiff(plot_path, width = 12, height = 8, units = "in", res = 300, compression = "lzw")
plot(cv_fit$glmnet.fit, xvar = "lambda", label = TRUE)
title("Coefficient Path")
abline(v = log(cv_fit$lambda.1se), col = "red", lty = 2, lwd = 1.5)
abline(v = log(cv_fit$lambda.min), col = "blue", lty = 2, lwd = 1.5)
dev.off()

tiff(plot_imp, width = 12, height = 8, units = "in", res = 300, compression = "lzw")
if (nrow(feats_1se) > 0) {
  cols <- ifelse(feats_1se$Coef > 0, "#E64B35", "#4DBBD5")
  par(mar = c(5, 12, 4, 2), xpd = NA)
  barplot(
    rev(feats_1se$Coef),
    names.arg = rev(feats_1se$Feature),
    horiz = TRUE,
    las = 1,
    col = rev(cols),
    border = "black",
    main = "Feature Importance",
    xlab = "Coefficient"
  )
  grid(ny = NA, col = "gray70", lty = "dotted")
} else {
  plot.new()
  title("Feature Importance")
  text(0.5, 0.5, "No non-zero features at lambda.1se")
}
dev.off()

cat("Saved:", outfile, "\n")
cat("Saved:", plot_cv, "\n")
cat("Saved:", plot_path, "\n")
cat("Saved:", plot_imp, "\n")
