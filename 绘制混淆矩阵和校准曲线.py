# =============================================================================
# Calibration Curve + Confusion Matrix (5 Models)
# Cohorts: Training / Internal_Val / External_Val
# Outputs:
# 1) Calibration plots (with LOWESS)
# 2) Confusion matrix plots
# 3) Calibration metrics table (Brier / ECE / calibration intercept & slope)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess


# -------------------------------
# 1. 读取数据
# -------------------------------
file_path = r"I:/RP_multib/重新修改/副本ALL911改11.xlsx"
df = pd.read_excel(file_path)

model_map = {
    "CAPRA": "pred_prob_CAPRA",
    "Clinical": "pred_prob_Clinical",
    "Image": "pred_prob_Image",
    "Habitat": "pred_prob_Habitat",
    "Combined": "pred_prob_Combined",
}

datasets = ["Training", "Internal_Val", "External_Val"]

required_cols = ["Dataset_Type", "AP_Status"] + list(model_map.values())
assert all(col in df.columns for col in required_cols), "❌ 缺少必要列"

# 仅保留可用结局
df["AP_Status"] = pd.to_numeric(df["AP_Status"], errors="coerce")
df = df[df["AP_Status"].isin([0, 1])].copy()
df["AP_Status"] = df["AP_Status"].astype(int)

for col in model_map.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")


# -------------------------------
# 2. 校准指标函数
# -------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)

    valid = (~np.isnan(y)) & (~np.isnan(p))
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        return np.nan

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins[1:-1], right=True)

    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) == 0:
            continue
        obs = np.mean(y[mask])
        pred = np.mean(p[mask])
        w = np.sum(mask) / n
        ece += w * abs(obs - pred)
    return float(ece)


def calibration_intercept_slope(y_true, y_prob):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)

    valid = (~np.isnan(y)) & (~np.isnan(p))
    y = y[valid]
    p = p[valid]

    if len(y) == 0 or np.unique(y).size < 2:
        return np.nan, np.nan

    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    X = sm.add_constant(lp)

    try:
        fit = sm.Logit(y, X).fit(disp=False)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def calc_calibration_metrics(y_true, y_prob):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)

    valid = (~np.isnan(y)) & (~np.isnan(p))
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        return {
            "N": 0,
            "Events": 0,
            "Brier": np.nan,
            "ECE": np.nan,
            "Cal_Intercept": np.nan,
            "Cal_Slope": np.nan,
        }

    brier = brier_score_loss(y, p)
    ece = expected_calibration_error(y, p, n_bins=10)
    cal_intercept, cal_slope = calibration_intercept_slope(y, p)

    return {
        "N": int(len(y)),
        "Events": int(np.sum(y)),
        "Brier": float(brier),
        "ECE": float(ece),
        "Cal_Intercept": cal_intercept,
        "Cal_Slope": cal_slope,
    }


# -------------------------------
# 3. 绘图函数
# -------------------------------
def plot_calibration(ax, y_true, y_prob, model_name, dataset_name):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    valid = (~np.isnan(y)) & (~np.isnan(p))
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        ax.set_title(f"{model_name} ({dataset_name})\nNo valid data", fontsize=8)
        ax.axis("off")
        return

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    ax.plot(
        mean_pred,
        frac_pos,
        marker="o",
        color="black",
        linewidth=0,
        markersize=3.5,
        alpha=0.8,
        label="Binned observed",
    )

    if len(mean_pred) >= 4:
        order = np.argsort(mean_pred)
        x_sorted = mean_pred[order]
        y_sorted = frac_pos[order]
        smooth = lowess(y_sorted, x_sorted, frac=0.6, it=0, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color="black", linewidth=2, label="LOWESS")
    else:
        ax.plot(mean_pred, frac_pos, color="black", linewidth=2, label="Curve")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", fontsize=8)
    ax.set_ylabel("Observed probability", fontsize=8)
    ax.set_title(f"{model_name} ({dataset_name})", fontsize=8, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def plot_confusion(ax, y_true, y_prob, model_name, dataset_name, threshold=0.5):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    valid = (~np.isnan(y)) & (~np.isnan(p))
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        ax.set_title(f"{model_name} ({dataset_name})\nNo valid data", fontsize=8)
        ax.axis("off")
        return

    y_pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.8,
        ax=ax,
        annot_kws={"size": 10},
    )

    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_title(f"{model_name} ({dataset_name})", fontsize=8, weight="bold")
    ax.set_xticklabels(["AP-", "AP+"], fontsize=8)
    ax.set_yticklabels(["AP-", "AP+"], rotation=90, va="center", fontsize=8)

    ax.hlines(y=-0.5, xmin=-0.5, xmax=1.5, color="black", linewidth=1.2)
    ax.vlines(x=-0.5, ymin=-0.5, ymax=1.5, color="black", linewidth=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -------------------------------
# 4. 批量绘制与指标汇总
# -------------------------------
metrics_rows = []

for model_name, pred_col in model_map.items():
    # 4.1 校准曲线：每个模型输出一个 1x3 图（Training / Internal / External）
    fig_cal, axes_cal = plt.subplots(1, 3, figsize=(9, 3), dpi=300)

    # 4.2 混淆矩阵：每个模型输出一个 1x3 图（Training / Internal / External）
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(9, 3), dpi=300)

    for i, ds_name in enumerate(datasets):
        ds = df[df["Dataset_Type"] == ds_name].copy()

        y = ds["AP_Status"].values
        p = ds[pred_col].values

        plot_calibration(axes_cal[i], y, p, model_name, ds_name)
        plot_confusion(axes_cm[i], y, p, model_name, ds_name, threshold=0.5)

        m = calc_calibration_metrics(y, p)
        metrics_rows.append(
            {
                "Model": model_name,
                "Dataset": ds_name,
                **m,
            }
        )

    fig_cal.tight_layout()
    fig_cm.tight_layout()

    fig_cal.savefig(
        f"Calibration_{model_name}_Training_Internal_External.tiff",
        dpi=600,
        format="tiff",
    )
    fig_cm.savefig(
        f"ConfusionMatrix_{model_name}_Training_Internal_External.tiff",
        dpi=600,
        format="tiff",
    )

    plt.show()


# -------------------------------
# 5. 保存校准定量指标
# -------------------------------
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_excel(
    "Calibration_Metrics_5Models_Training_Internal_External.xlsx",
    index=False,
)

print("✅ 已完成：5模型在Training/Internal_Val/External_Val的校准曲线与混淆矩阵绘制")
print("✅ 已保存校准定量指标：Calibration_Metrics_5Models_Training_Internal_External.xlsx")

# =============================================================================
# End of script
# =============================================================================
