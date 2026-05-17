# ==============================================================================
# 模型评估脚本（兼容 pROC 1.18.5，去掉雷达图，增加ROC曲线 + Δrisk + IDI可视化）
# ==============================================================================
rm(list=ls())
options(stringsAsFactors = FALSE, warn=-1)

# -------------------------------
# 0. 载入包
# -------------------------------
packages <- c("readxl","dplyr","pROC","rmda","ggplot2","openxlsx","DescTools",
              "nricens","scales","rms","PredictABEL","tidyr")
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

required_cols <- c("Dataset_Type","AP_Status","pred_prob_Clinical","pred_prob_ImageModel","pred_prob_Habitat",
                   "pred_prob_Combined","pred_prob_CAPRA")
if(!all(required_cols %in% names(df))){
  stop("❌ 数据缺少必要列，请检查")
}

# -------------------------------
# 6. Δrisk density + IDI bar plot（输出两版）
# -------------------------------
build_delta_plots <- function(data_in, dataset_levels, out_tag){
  df_plot <- data_in %>%
    mutate(
      Dataset_Type = trimws(Dataset_Type),
      Dataset_Type = factor(Dataset_Type, levels = dataset_levels),
      AP_Status = as.numeric(AP_Status)
    ) %>%
    filter(Dataset_Type %in% dataset_levels, AP_Status %in% c(0, 1)) %>%
    select(Dataset_Type, AP_Status,
           pred_prob_Combined,
           pred_prob_Clinical,
           pred_prob_ImageModel,
           pred_prob_Habitat,
           pred_prob_CAPRA) %>%
    mutate(
      d_Clinical = pred_prob_Combined - pred_prob_Clinical,
      d_Image    = pred_prob_Combined - pred_prob_ImageModel,
      d_Habitat  = pred_prob_Combined - pred_prob_Habitat,
      d_CAPRA    = pred_prob_Combined - pred_prob_CAPRA
    ) %>%
    pivot_longer(
      cols = starts_with("d_"),
      names_to = "Comparison",
      values_to = "DeltaRisk"
    ) %>%
    mutate(
      Comparison = factor(
        Comparison,
        levels = c("d_CAPRA", "d_Clinical", "d_Habitat", "d_Image"),
        labels = c("Combined vs CAPRA",
                   "Combined vs Clinical",
                   "Combined vs Habitat",
                   "Combined vs Image")
      ),
      Outcome = factor(AP_Status, levels = c(0, 1), labels = c("Non-AP", "AP"))
    )

  p_density <- ggplot(df_plot, aes(x = DeltaRisk, fill = Outcome)) +
    geom_density(alpha = 0.45, adjust = 1.2, color = "gray60", size = 0.4) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", size = 0.4) +
    facet_grid(Dataset_Type ~ Comparison, scales = "free_y") +
    scale_fill_manual(name = NULL, values = c("Non-AP" = "#4DBBD5", "AP" = "#E64B35")) +
    coord_cartesian(xlim = c(-0.6, 0.6)) +
    labs(x = expression(Delta~Risk~(Combined~-~Comparator)), y = "Density") +
    theme_bw(base_size = 14) +
    theme(
      strip.background = element_rect(fill = "grey95", color = NA),
      strip.text = element_text(size = 13, face = "bold"),
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", size = 0.5),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14),
      panel.spacing.x = unit(0.20, "cm"),
      legend.position = c(1, 1),
      legend.justification = c(1, 1),
      legend.key.size = unit(0.6, "cm"),
      legend.text = element_text(size = 9)
    )

  print(p_density)
  ggsave(
    paste0("I:/RP_multib/DeltaRisk_Density_", out_tag, ".pdf"),
    p_density,
    width = 8.5,
    height = ifelse(length(dataset_levels) == 3, 6.0, 4.5)
  )

  idi_df <- df_plot %>%
    group_by(Dataset_Type, Comparison, Outcome) %>%
    summarise(mean_delta = mean(DeltaRisk), .groups = "drop") %>%
    pivot_wider(names_from = Outcome, values_from = mean_delta) %>%
    mutate(IDI = AP - `Non-AP`)

  fill_map <- c("Training" = "#E69F00", "Internal_Val" = "#00A087", "External_Val" = "#3C5488")
  fill_map <- fill_map[names(fill_map) %in% dataset_levels]

  p_idi <- ggplot(idi_df, aes(x = Comparison, y = IDI, fill = Dataset_Type)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.62) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    labs(y = "Integrated Discrimination Improvement (IDI)", x = NULL) +
    scale_fill_manual(values = fill_map) +
    theme_bw(base_size = 14) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(size = 12, angle = 15, hjust = 1),
      axis.text.y = element_text(size = 12),
      axis.title.y = element_text(size = 14),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = 12)
    )

  print(p_idi)
  ggsave(
    paste0("I:/RP_multib/IDI_Bar_", out_tag, ".pdf"),
    p_idi,
    width = 7.2,
    height = 4.4
  )
}

# 版本1：Training + Internal_Val + External_Val
build_delta_plots(
  data_in = df,
  dataset_levels = c("Training", "Internal_Val", "External_Val"),
  out_tag = "Training_Internal_External"
)

# 版本2：Internal_Val + External_Val
build_delta_plots(
  data_in = df,
  dataset_levels = c("Internal_Val", "External_Val"),
  out_tag = "Internal_External"
)

cat("\n✅ Δrisk/IDI可视化完成：已输出三队列版与验证集版，且字体已调大\n")
