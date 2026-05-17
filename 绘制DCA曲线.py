import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"I:/RP_multib/重新修改/副本ALL911改11.xlsx"
df = pd.read_excel(file_path)

MODEL_MAP = {
    "CAPRA": "pred_prob_CAPRA",
    "Clinical": "pred_prob_Clinical",
    "Image": "pred_prob_Image",
    "Habitat": "pred_prob_Habitat",
    "Combined": "pred_prob_Combined",
}

required_cols = ["Dataset_Type", "AP_Status"] + list(MODEL_MAP.values())
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要列: {col}")

df["AP_Status"] = pd.to_numeric(df["AP_Status"], errors="coerce")
df = df[df["AP_Status"].isin([0, 1])].copy()
df["AP_Status"] = df["AP_Status"].astype(int)

for col in MODEL_MAP.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")


def net_benefit(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    valid = (~np.isnan(y_true)) & (~np.isnan(y_prob))
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    N = len(y_true)
    if N == 0:
        return np.full_like(thresholds, np.nan, dtype=float)

    nb = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb_val = (tp / N) - (fp / N) * (pt / (1 - pt))
        nb.append(nb_val)
    return np.array(nb)


def plot_dca(ds_df, dataset_name, save_path=None, fig_size=(5, 5), dpi=600):
    thresholds = np.linspace(0.01, 0.99, 120)
    y = ds_df["AP_Status"].values

    nb = {
        "CAPRA": net_benefit(y, ds_df[MODEL_MAP["CAPRA"]].values, thresholds),
        "Clinical": net_benefit(y, ds_df[MODEL_MAP["Clinical"]].values, thresholds),
        "Image": net_benefit(y, ds_df[MODEL_MAP["Image"]].values, thresholds),
        "Habitat": net_benefit(y, ds_df[MODEL_MAP["Habitat"]].values, thresholds),
        "Combined": net_benefit(y, ds_df[MODEL_MAP["Combined"]].values, thresholds),
    }

    prevalence = np.nanmean(y)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_none = np.zeros_like(thresholds)

    plt.figure(figsize=fig_size)
    markevery = 12
    styles = {
        "CAPRA": dict(color="#1F77B4", lw=1.8, ls="-", marker="o"),
        "Clinical": dict(color="#2CA02C", lw=1.8, ls="--", marker="s"),
        "Image": dict(color="#D62728", lw=1.8, ls="-.", marker="^"),
        "Habitat": dict(color="#FF7F0E", lw=1.8, ls=":", marker="D"),
        "Combined": dict(color="black", lw=2.6, ls="-", marker=None),
    }

    for name in ["CAPRA", "Clinical", "Image", "Habitat", "Combined"]:
        st = styles[name]
        plt.plot(
            thresholds,
            nb[name],
            color=st["color"],
            lw=st["lw"],
            ls=st["ls"],
            marker=st["marker"],
            ms=3 if st["marker"] else None,
            markevery=markevery if st["marker"] else None,
            label=name,
        )

    plt.plot(thresholds, nb_all, color="grey", lw=1.2, ls="--", alpha=0.8, label="Treat all")
    plt.plot(thresholds, nb_none, color="grey", lw=1.2, ls=":", alpha=0.8, label="Treat none")

    ax = plt.gca()
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.xlabel("Threshold Probability", fontsize=13)
    plt.ylabel("Net Benefit", fontsize=13)
    plt.title(f"Decision Curve Analysis ({dataset_name})", fontsize=13, weight="bold")
    ax.tick_params(axis="both", labelsize=12)
    plt.legend(loc="lower left", fontsize=11, frameon=False)
    plt.xlim(0.01, 0.99)
    plt.ylim(0, 0.65)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def clinical_impact_curve(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    valid = (~np.isnan(y_true)) & (~np.isnan(y_prob))
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    n = len(y_true)
    if n == 0:
        return np.full_like(thresholds, np.nan), np.full_like(thresholds, np.nan)

    high_risk = []
    true_events = []

    for pt in thresholds:
        pred_high = y_prob >= pt
        n_high = pred_high.sum()
        n_true_event_high = np.sum(pred_high & (y_true == 1))

        high_risk.append(n_high / n * 1000)
        true_events.append(n_true_event_high / n * 1000)

    return np.array(high_risk), np.array(true_events)


def plot_clinical_impact(ds_df, dataset_name, save_path=None, fig_size=(7, 5), dpi=600):
    thresholds = np.linspace(0.01, 0.99, 120)
    y = ds_df["AP_Status"].values

    styles = {
        "CAPRA": dict(color="#1F77B4", ls="-"),
        "Clinical": dict(color="#2CA02C", ls="--"),
        "Image": dict(color="#D62728", ls="-."),
        "Habitat": dict(color="#FF7F0E", ls=":"),
        "Combined": dict(color="black", ls="-"),
    }

    plt.figure(figsize=fig_size)

    for model_name in ["CAPRA", "Clinical", "Image", "Habitat", "Combined"]:
        high_risk, true_events = clinical_impact_curve(y, ds_df[MODEL_MAP[model_name]].values, thresholds)

        plt.plot(
            thresholds,
            high_risk,
            color=styles[model_name]["color"],
            ls=styles[model_name]["ls"],
            lw=1.8,
            alpha=0.9,
            label=f"{model_name} high risk",
        )
        plt.plot(
            thresholds,
            true_events,
            color=styles[model_name]["color"],
            ls=styles[model_name]["ls"],
            lw=1.2,
            alpha=0.45,
            label=f"{model_name} true events",
        )

    ax = plt.gca()
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel("Threshold Probability", fontsize=13)
    plt.ylabel("Number per 1000", fontsize=13)
    plt.title(f"Clinical Impact Curve ({dataset_name})", fontsize=13, weight="bold")
    ax.tick_params(axis="both", labelsize=12)
    plt.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    plt.xlim(0.01, 0.99)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_reclassification_heatmap(ds_df, dataset_name, old_model="Clinical", new_model="Combined", save_path=None, dpi=600):
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    sub = ds_df[["AP_Status", MODEL_MAP[old_model], MODEL_MAP[new_model]]].dropna().copy()
    if sub.empty:
        return

    sub["old_bin"] = pd.cut(sub[MODEL_MAP[old_model]], bins=bins, labels=labels, include_lowest=True)
    sub["new_bin"] = pd.cut(sub[MODEL_MAP[new_model]], bins=bins, labels=labels, include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    for ax, event_value, title_suffix in zip(axes, [1, 0], ["Events (AP+)", "Non-events (AP-)"]):
        tmp = sub[sub["AP_Status"] == event_value]
        tab = pd.crosstab(tmp["old_bin"], tmp["new_bin"], normalize="index") * 100
        tab = tab.reindex(index=labels, columns=labels)

        sns.heatmap(
            tab,
            ax=ax,
            cmap="YlGnBu",
            annot=True,
            fmt=".1f",
            cbar=event_value == 1,
            vmin=0,
            vmax=100,
            linewidths=0.5,
        )
        ax.set_xlabel(f"{new_model} risk group")
        ax.set_ylabel(f"{old_model} risk group")
        ax.set_title(f"{title_suffix}")

    fig.suptitle(f"Reclassification Heatmap ({dataset_name}): {old_model} -> {new_model}", fontsize=12, weight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_probability_distribution(ds_df, dataset_name, save_path=None, dpi=600):
    long_rows = []
    for model_name, col in MODEL_MAP.items():
        tmp = ds_df[["AP_Status", col]].dropna().copy()
        tmp = tmp.rename(columns={col: "PredProb"})
        tmp["Model"] = model_name
        tmp["Outcome"] = tmp["AP_Status"].map({0: "AP-", 1: "AP+"})
        long_rows.append(tmp[["Model", "Outcome", "PredProb"]])

    long_df = pd.concat(long_rows, ignore_index=True)

    plt.figure(figsize=(9, 5), dpi=300)
    sns.violinplot(
        data=long_df,
        x="Model",
        y="PredProb",
        hue="Outcome",
        split=True,
        inner="quartile",
        linewidth=1,
        palette={"AP-": "#4C78A8", "AP+": "#E45756"},
    )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ylim(0, 1)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Predicted probability", fontsize=12)
    plt.title(f"Predicted Probability Distribution ({dataset_name})", fontsize=13, weight="bold")
    plt.legend(title="Outcome", frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


# ===============================
# 执行：Training / Internal / External
# ===============================
datasets = {
    "Training": df[df["Dataset_Type"] == "Training"].copy(),
    "Internal_Val": df[df["Dataset_Type"] == "Internal_Val"].copy(),
    "External_Val": df[df["Dataset_Type"] == "External_Val"].copy(),
}

for ds_name, ds_df in datasets.items():
    if ds_df.empty:
        continue

    plot_dca(ds_df, ds_name, save_path=f"DCA_{ds_name}.tiff")
    '''
    plot_clinical_impact(ds_df, ds_name, save_path=f"ClinicalImpact_{ds_name}.tiff")

    plot_reclassification_heatmap(
        ds_df,
        dataset_name=ds_name,
        old_model="Clinical",
        new_model="Combined",
        save_path=f"ReclassHeatmap_Clinical_to_Combined_{ds_name}.tiff",
    )

    plot_reclassification_heatmap(
        ds_df,
        dataset_name=ds_name,
        old_model="Image",
        new_model="Combined",
        save_path=f"ReclassHeatmap_Image_to_Combined_{ds_name}.tiff",
    )

    plot_probability_distribution(
        ds_df,
        dataset_name=ds_name,
        save_path=f"ProbabilityDistribution_5Models_{ds_name}.tiff",
    )
'''
print("✅ DCA、临床影响曲线、重分类热图、概率分布图已全部生成")
