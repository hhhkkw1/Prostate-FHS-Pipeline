import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)

# ===============================
# 1. 读取数据
# ===============================
file_path = r"I:/RP_multib/重新修改/副本ALL911改11.xlsx"
df = pd.read_excel(file_path)

models = {
    "CAPRA": "pred_prob_CAPRA",
    "Clinical": "pred_prob_Clinical",
    "Habitat": "pred_prob_Habitat",
    "Combined": "pred_prob_Combined"
}

outcome = "AP_Status"
dataset_col = "Dataset_Type"

# ===============================
# 2. 计算模型指标（阈值 = 0.5）
# ===============================
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)

    return [sensitivity, specificity, ppv, npv, accuracy]

# 雷达图指标名称（顺序很重要）
metrics_name = ["Sensitivity", "Specificity", "PPV", "NPV", "Accuracy"]

# ===============================
# 3. 计算 Internal / External 指标
# ===============================
results = {
    "Internal_Val": {},
    "External_Val": {}
}

for ds in results.keys():
    df_ds = df[df[dataset_col] == ds]
    y_true = df_ds[outcome].values

    for model, col in models.items():
        y_prob = df_ds[col].values
        results[ds][model] = compute_metrics(y_true, y_prob)

# ===============================
# 4. 雷达图绘制函数（无填充）
# ===============================
def plot_radar(performance, title, save_name):
    labels = metrics_name
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = {
        "Combined": "black",
        "CAPRA": "#1F77B4",
        "Clinical": "#2CA02C",
        "Habitat": "#FF7F0E"
    }

    line_widths = {
        "Combined": 1.5,
        "CAPRA": 1,
        "Clinical": 1,
        "Habitat": 1
    }

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)

    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # 只画线，不填充
    for model in ["CAPRA", "Clinical", "Habitat","Combined"]:
        values = performance[model] + performance[model][:1]
        ax.plot(
            angles,
            values,
            color=colors[model],
            linewidth=line_widths[model],
            label=model
        )

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.92, 0.03),
        frameon=False,
        fontsize=10
    )

    ax.set_title(title, fontsize=12, weight="bold")

    plt.tight_layout()
    plt.savefig(save_name, dpi=600, format="tiff")
    plt.show()

# ===============================
# 5. 绘制雷达图
# ===============================
plot_radar(
    results["Internal_Val"],
    title="Radar Chart (Internal_Val)",
    save_name="Radar_Internal.tiff"
)

plot_radar(
    results["External_Val"],
    title="Radar Chart (External_Val)",
    save_name="Radar_External.tiff"
)