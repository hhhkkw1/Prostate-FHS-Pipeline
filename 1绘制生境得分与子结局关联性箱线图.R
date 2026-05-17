# ==============================================================================
# 0. 环境设置
# ==============================================================================
rm(list = ls()) 
options(warn = -1)

library(readxl)
library(dplyr)
library(ggplot2)
library(ggpubr)   
library(gridExtra)
library(tibble)

# ==============================================================================
# 1. 加载数据
# ==============================================================================
FILE_PATH <- "I:/RP_multib/重新修改/副本ALL911改11.xlsx"

cat("📂 正在读取数据...\n")
df <- read_excel(FILE_PATH)

# 确保 Habitat_score 存在 (如果只保存了 Rad_score_1se，则用它)
if (!"Habitat_score" %in% names(df)) {
  if ("Rad_score_1se" %in% names(df)) {
    df$Habitat_score <- df$Rad_score_1se
  } else {
    stop("❌ 找不到 Habitat_score 或 Rad_score_1se 列！")
  }
}

# 确保 AP_Status 存在
sub_outcomes_calc <- c('EPE', 'SVI', 'Margin', 'High_GG')
df$Lymph_Calc <- ifelse(is.na(df$Lymph), 0, df$Lymph)
df$AP_Status <- as.integer(rowSums(df[, c(sub_outcomes_calc, "Lymph_Calc")], na.rm = TRUE) > 0)

# ==============================================================================
# 2. 定义 6 个结局变量 (AP放第一个)
# ==============================================================================
# 顺序：AP -> EPE -> SVI -> Margin -> High_GG -> Lymph
target_outcomes <- c('AP_Status', 'EPE', 'SVI', 'Margin', 'High_GG', 'Lymph')

# 定义漂亮的标题映射
outcome_labels <- list(
  'AP_Status' = 'Total Adverse Pathology',  # 总结局
  'EPE'       = 'Extraprostatic Extension',
  'SVI'       = 'Seminal Vesicle Invasion',
  'Margin'    = 'Positive Surgical Margin',
  'High_GG'   = 'High Grade (ISUP >= 4)',
  'Lymph'     = 'Lymph Node Invasion'
)

# ==============================================================================
# 3. FDR 多重校正 (BH)
# ==============================================================================
cat("🧪 正在计算 Wilcoxon 检验并进行 FDR(BH) 校正...\n")

test_results <- lapply(target_outcomes, function(outcome_col) {
  if (!outcome_col %in% names(df)) return(NULL)
  
  tmp <- df %>%
    select(all_of(c(outcome_col, "Habitat_score"))) %>%
    na.omit() %>%
    rename(Outcome = !!outcome_col, Score = Habitat_score)
  
  if (length(unique(tmp$Outcome)) < 2) {
    return(tibble(
      Outcome = outcome_col,
      p_raw = NA_real_
    ))
  }
  
  p_raw <- tryCatch(
    wilcox.test(Score ~ Outcome, data = tmp)$p.value,
    error = function(e) NA_real_
  )
  
  tibble(
    Outcome = outcome_col,
    p_raw = p_raw
  )
})

test_df <- bind_rows(test_results)
test_df <- test_df %>%
  mutate(
    p_fdr = p.adjust(p_raw, method = "BH"),
    signif_label = case_when(
      is.na(p_fdr) ~ "ns",
      p_fdr < 0.001 ~ "***",
      p_fdr < 0.01 ~ "**",
      p_fdr < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

print(test_df)

# ==============================================================================
# 4. 绘图函数 (小提琴 + 窄箱线图)
# ==============================================================================
create_violin <- function(data, outcome_col, star_label, score_col = "Habitat_score") {
  
  # 提取数据并清洗
  plot_data <- data %>%
    select(all_of(c(outcome_col, score_col))) %>%
    na.omit() %>%
    rename(Outcome = !!outcome_col, Score = !!score_col) %>%
    mutate(Outcome = factor(Outcome, levels = c(0, 1), labels = c("Negative", "Positive")))
  
  # 统计人数
  n_neg <- sum(plot_data$Outcome == "Negative")
  n_pos <- sum(plot_data$Outcome == "Positive")
  
  # 绘图
  p <- ggplot(plot_data, aes(x = Outcome, y = Score)) +
    
    # Layer 1: 小提琴图 (展示密度分布)
    # trim=FALSE 让尾部平滑延伸，alpha=0.6 设置透明度
    geom_violin(aes(fill = Outcome, color = Outcome), 
                trim = FALSE, alpha = 0.6, width = 0.5, size = 0.5) +
    
    # Layer 2: 内部窄箱线图 (展示统计量)
    # width=0.1 很窄，fill="white" 让它在中间显眼
    geom_boxplot(width = 0.15, fill = "white", color = "black", 
                 outlier.shape = NA, alpha = 0.9) +
    
    # Layer 3: 使用FDR校正后的星号标注（保持原有“星号表示显著性”的逻辑）
    annotate("text", x = 1.5, y = max(plot_data$Score) * 1.05,
             label = star_label, size = 6, fontface = "bold", vjust = 0) +
    
    # 美化设置
    scale_fill_manual(values = c("#4DBBD5", "#E64B35")) +  # 填充色: 蓝 vs 红
    scale_color_manual(values = c("#4DBBD5", "#E64B35")) + # 边框色同填充色
    theme_pubr(base_size = 14) +
    labs(title = outcome_labels[[outcome_col]], 
         x = NULL, y = "FHS") +
    
    # X轴标签带样本量
    scale_x_discrete(labels = c(paste0("Neg"), 
                                paste0("Pos"))) +
    
    theme(legend.position = "none",
          plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
          axis.text = element_text(size = 10),
          axis.title.y = element_text(size = 10))
  
  return(p)
}

# ==============================================================================
# 5. 批量生成并拼图
# ==============================================================================
cat("📊 正在生成小提琴图 (N=6)...\n")

plot_list <- list()

for (outcome in target_outcomes) {
  if (outcome %in% names(df)) {
    star_label <- test_df %>%
      filter(Outcome == outcome) %>%
      pull(signif_label)
    
    if (length(star_label) == 0) star_label <- "ns"
    
    p <- create_violin(df, outcome, star_label = star_label)
    plot_list[[outcome]] <- p
  }
}

# 拼接图形 (2行3列)
final_plot <- ggarrange(plotlist = plot_list, 
                        ncol = 3, nrow = 2)

# 添加总标题
#final_plot <- annotate_figure(final_plot,
 #                             top = text_grob("", 
  #                                            color = "black", face = "bold", size = 15))

# ==============================================================================
# 6. 保存结果
# ==============================================================================
SAVE_PATH <- gsub(".xlsx", "_Violin_Plots_6Panels2.tiff", FILE_PATH)
ggsave(SAVE_PATH, final_plot, width = 7, height = 5)

# 额外导出统计结果表
TEST_SAVE_PATH <- gsub(".xlsx", "_Violin_Wilcox_FDR_Results.csv", FILE_PATH)
write.csv(test_df, TEST_SAVE_PATH, row.names = FALSE)

cat(sprintf("✅ 完成！图片已保存至: %s\n", SAVE_PATH))
cat(sprintf("✅ FDR校正结果已保存至: %s\n", TEST_SAVE_PATH))
