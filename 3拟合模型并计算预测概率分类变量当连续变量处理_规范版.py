# ==============================================================================
# 0. 环境准备
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


# ==============================================================================
# 1. 读取数据
# ==============================================================================
FILE_PATH = r"I:\RP_multib\重新修改\副本ALL911改11.xlsx"

print(f"📂 正在读取数据: {FILE_PATH}")
df = pd.read_excel(FILE_PATH)


# ==============================================================================
# 2. 检查必要列
# ==============================================================================
required_cols = [
    "Dataset_Type",
    "AP_Status",
    "age",
    "PSA",
    "T_score",
    "GG_BP",
    "PIRADS",
    "pct_pos",
    "Habitat_score",
    "CAPRA_score",
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ 缺少列: {col}")


# ==============================================================================
# 3. 基础类型处理
# ==============================================================================
print("🧹 正在进行基础数据处理...")

continuous_cols = [
    "age",
    "PSA",
    "pct_pos",
    "Habitat_score",
    "CAPRA_score",
    "T_score",
    "GG_BP",
    "PIRADS",
]

for col in continuous_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["AP_Status"] = pd.to_numeric(df["AP_Status"], errors="coerce")


# ==============================================================================
# 4. 划分训练集 + 结局缺失处理
# ==============================================================================
train_mask = df["Dataset_Type"].astype(str).str.strip().str.lower() == "training"
train_df = df.loc[train_mask].copy()

# 训练数据中结局缺失必须剔除，避免把未知结局当阴性
train_df = train_df[train_df["AP_Status"].notna()].copy()

# 仅保留0/1结局
train_df = train_df[train_df["AP_Status"].isin([0, 1])].copy()
train_df["AP_Status"] = train_df["AP_Status"].astype(int)

y_train = train_df["AP_Status"]

if train_df.empty:
    raise ValueError("❌ Training集中无可用样本（AP_Status需为0/1且非缺失）")

if y_train.nunique() < 2:
    raise ValueError("❌ Training集中结局仅有单一类别，无法拟合Logistic回归")


# ==============================================================================
# 5. 缺失值处理（防信息泄露）
# ==============================================================================
print("🛡️ 正在进行防泄露缺失值填充（仅用Training统计量）...")

train_medians = train_df[continuous_cols].median()
df[continuous_cols] = df[continuous_cols].fillna(train_medians)

# 填充后同步更新训练集
train_df = df.loc[train_mask].copy()
train_df = train_df[train_df["AP_Status"].notna() & df.loc[train_mask, "AP_Status"].isin([0, 1])].copy()
train_df["AP_Status"] = train_df["AP_Status"].astype(int)
y_train = train_df["AP_Status"]

print("✅ 缺失值处理完成")


# ==============================================================================
# 6. 连续变量形式建模
# ==============================================================================
features_clin = ["age", "PSA", "pct_pos", "T_score", "GG_BP"]
features_image = ["PIRADS"]
features_habitat = ["Habitat_score"]
features_comb = features_clin + features_habitat + features_image
features_capra = ["CAPRA_score"]


# ==============================================================================
# 7. LogisticRegression建模
# ==============================================================================
print("🏗️ 正在训练模型...")

lr_clin = LogisticRegression(max_iter=1000, solver="liblinear")
lr_image = LogisticRegression(max_iter=1000, solver="liblinear")
lr_hab = LogisticRegression(max_iter=1000, solver="liblinear")
lr_comb = LogisticRegression(max_iter=1000, solver="liblinear")
lr_capra = LogisticRegression(max_iter=1000, solver="liblinear")

lr_clin.fit(train_df[features_clin], y_train)
lr_image.fit(train_df[features_image], y_train)
lr_hab.fit(train_df[features_habitat], y_train)
lr_comb.fit(train_df[features_comb], y_train)
lr_capra.fit(train_df[features_capra], y_train)

print("✅ 模型训练完成")


# ==============================================================================
# 8. 预测概率
# ==============================================================================
print("🔮 正在生成预测概率...")

df["pred_prob_Clinical"] = lr_clin.predict_proba(df[features_clin])[:, 1]
df["pred_prob_Image"] = lr_image.predict_proba(df[features_image])[:, 1]
df["pred_prob_Habitat"] = lr_hab.predict_proba(df[features_habitat])[:, 1]
df["pred_prob_Combined"] = lr_comb.predict_proba(df[features_comb])[:, 1]
df["pred_prob_CAPRA"] = lr_capra.predict_proba(df[features_capra])[:, 1]


# ==============================================================================
# 9. 保存预测结果
# ==============================================================================
OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_Predictions_Continuous_规范版.xlsx")
print(f"💾 正在保存预测结果: {OUTPUT_PATH}")
df.to_excel(OUTPUT_PATH, index=False)
print("✅ 预测结果保存完成")


# ==============================================================================
# 10. 提取 Logistic 回归参数
# ==============================================================================
def extract_logistic_params(X, y, model_name):
    print(f"\n📊 正在计算 {model_name} Logistic参数")

    X = X.copy().astype(float)
    y = y.copy().astype(int)

    X_sm = sm.add_constant(X)
    model = sm.Logit(y, X_sm)
    result = model.fit(disp=False)

    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    summary_df = pd.DataFrame(
        {
            "Model": model_name,
            "Variable": params.index,
            "Beta": params.values,
            "OR": np.exp(params.values),
            "CI_lower": np.exp(conf[0].values),
            "CI_upper": np.exp(conf[1].values),
            "p_value": pvals.values,
        }
    )

    summary_df = summary_df[summary_df["Variable"] != "const"]
    return summary_df


# ==============================================================================
# 11. 计算AUC / Brier / 校准
# ==============================================================================
def calibration_intercept_slope(y_true, p_pred):
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=int)

    lp = np.log(p / (1 - p))
    X = sm.add_constant(lp)

    try:
        fit = sm.Logit(y, X).fit(disp=False)
        intercept = float(fit.params[0])
        slope = float(fit.params[1])
    except Exception:
        intercept = np.nan
        slope = np.nan

    return intercept, slope


def evaluate_predictions(eval_df, pred_col, model_name, group_name):
    sub = eval_df[["AP_Status", pred_col]].dropna().copy()
    sub = sub[sub["AP_Status"].isin([0, 1])]

    n = len(sub)
    if n == 0:
        return {
            "Model": model_name,
            "Dataset_Group": group_name,
            "N": 0,
            "Events": 0,
            "AUC": np.nan,
            "Brier": np.nan,
            "Cal_Intercept": np.nan,
            "Cal_Slope": np.nan,
        }

    y = sub["AP_Status"].astype(int).values
    p = sub[pred_col].astype(float).values

    events = int(y.sum())

    if np.unique(y).size < 2:
        auc = np.nan
    else:
        auc = float(roc_auc_score(y, p))

    brier = float(brier_score_loss(y, p))
    cal_intercept, cal_slope = calibration_intercept_slope(y, p)

    return {
        "Model": model_name,
        "Dataset_Group": group_name,
        "N": int(n),
        "Events": events,
        "AUC": auc,
        "Brier": brier,
        "Cal_Intercept": cal_intercept,
        "Cal_Slope": cal_slope,
    }


def build_metrics_table(all_df):
    eval_df = all_df[all_df["AP_Status"].isin([0, 1])].copy()

    model_map = {
        "Clinical": "pred_prob_Clinical",
        "Image": "pred_prob_Image",
        "Habitat": "pred_prob_Habitat",
        "Combined": "pred_prob_Combined",
        "CAPRA": "pred_prob_CAPRA",
    }

    groups = {
        "All_Valid": eval_df,
        "Training": eval_df[eval_df["Dataset_Type"].astype(str).str.strip().str.lower() == "training"],
        "Non_Training": eval_df[eval_df["Dataset_Type"].astype(str).str.strip().str.lower() != "training"],
    }

    rows = []
    for model_name, pred_col in model_map.items():
        for group_name, group_df in groups.items():
            rows.append(evaluate_predictions(group_df, pred_col, model_name, group_name))

    return pd.DataFrame(rows)


# ==============================================================================
# 12. 汇总并保存参数 + 性能结果
# ==============================================================================
params_all = [
    extract_logistic_params(train_df[features_clin], y_train, "Clinical"),
    extract_logistic_params(train_df[features_image], y_train, "Image"),
    extract_logistic_params(train_df[features_habitat], y_train, "Habitat"),
    extract_logistic_params(train_df[features_comb], y_train, "Combined"),
    extract_logistic_params(train_df[features_capra], y_train, "CAPRA"),
]

params_df = pd.concat(params_all, ignore_index=True)
metrics_df = build_metrics_table(df)

PARAM_OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_Logistic_Params_Continuous_规范版.xlsx")
print(f"💾 正在保存参数与性能结果: {PARAM_OUTPUT_PATH}")

with pd.ExcelWriter(PARAM_OUTPUT_PATH, engine="openpyxl") as writer:
    params_df.to_excel(writer, sheet_name="Logistic_Params", index=False)
    metrics_df.to_excel(writer, sheet_name="Model_Performance", index=False)

print("✅ Logistic参数与模型性能保存完成")


# ==============================================================================
# 13. 输出摘要
# ==============================================================================
print("\n================ Logistic Parameters ================\n")
print(params_df)

print("\n================ Model Performance ================\n")
print(metrics_df)
