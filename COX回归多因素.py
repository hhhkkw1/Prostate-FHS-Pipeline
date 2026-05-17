# =============================================================================
# Cox Model Comparison
# C-index (Bootstrap P)
# Time-dependent AUC (Bar Plot)
# Export HR / CI / P for Combined Model
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample

from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 数据读取
# =============================================================================
FILE_PATH = "I:\RP_multib\重新修改\副本ALL911改11.xlsx"
OUT_EXCEL = "I:\RP_multib\重新修改\副本ALL911改11TimeDependent_AUC_Results.xlsx"

df = pd.read_excel(FILE_PATH)
df_surv = df.dropna(subset=["BCR", "BCRmonth"]).copy()

# =============================================================================
# 1. 数据预处理
# =============================================================================
cols = [
    "BCRmonth", "BCR",
    "PSA", "GG_BP", "Habitat_score",
    "CAPRA_score", "T_score", "age", "pct_pos"
]

for c in cols:
    df_surv[c] = pd.to_numeric(df_surv[c], errors="coerce")

df_train_raw = df_surv[df_surv["Dataset_Type"] == "Training"]
df_surv[cols] = df_surv[cols].fillna(df_train_raw[cols].median())

df_train = df_surv[df_surv["Dataset_Type"] == "Training"].copy()
df_internal = df_surv[df_surv["Dataset_Type"] == "Internal_Val"].copy()
df_external = df_surv[df_surv["Dataset_Type"] == "External_Val"].copy()

# =============================================================================
# 2. 模型定义
# =============================================================================
combined_vars = ["Habitat_score", "T_score", "PSA", "GG_BP", "age", "pct_pos", 'PIRADS']
capra_vars = ["CAPRA_score"]

cph_comb = CoxPHFitter()
cph_capra = CoxPHFitter()

cph_comb.fit(df_train[combined_vars + ["BCRmonth", "BCR"]],
             duration_col="BCRmonth",
             event_col="BCR")

cph_capra.fit(df_train[capra_vars + ["BCRmonth", "BCR"]],
              duration_col="BCRmonth",
              event_col="BCR")

# =============================================================================
# 3. 导出 Combined 模型 HR / CI / P
# =============================================================================
summary = cph_comb.summary.copy()
summary["HR"] = np.exp(summary["coef"])
summary["CI_lower"] = np.exp(summary["coef"] - 1.96 * summary["se(coef)"])
summary["CI_upper"] = np.exp(summary["coef"] + 1.96 * summary["se(coef)"])

hr_table = summary[["HR", "CI_lower", "CI_upper", "p"]]
print("\n=== Combined Model HR (95% CI) ===")
print(hr_table)

# =============================================================================
# 4. C-index + Bootstrap P-value
# =============================================================================
def get_c_index(model, data):
    return concordance_index(
        data["BCRmonth"],
        -model.predict_partial_hazard(data),
        data["BCR"]
    )

def bootstrap_pvalue(model_base, model_new, data, n=3000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    idx = np.arange(len(data))

    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = data.iloc[s]
        if d["BCR"].sum() < 2:
            continue
        diffs.append(
            get_c_index(model_new, d) -
            get_c_index(model_base, d)
        )

    diffs = np.array(diffs)
    return np.mean(diffs <= 0)


print("\n=== C-index Comparison ===")
plt.rcParams.update({
    "font.size": 12,          # 全局基础字体
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

for name, data in [
    ("Training", df_train),
    ("Internal", df_internal),
    ("External", df_external),
]:
    c1 = get_c_index(cph_capra, data)
    c2 = get_c_index(cph_comb, data)
    p = bootstrap_pvalue(cph_capra, cph_comb, data)
    print(f"{name:10s} | CAPRA={c1:.3f} | Combined={c2:.3f} | Δ={c2-c1:+.3f} | P={p:.4f}")

# =============================================================================
# 5. Time-dependent AUC + 95% CI + DeLong significance
# =============================================================================
from scipy import stats

times = np.array([12, 24, 36, 48, 60])
N_BOOT = 1000

# ---------- DeLong implementation ----------
def delong_roc_test(y_true, pred1, pred2):
    """
    Paired DeLong test, return p-value
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)

    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N)
        T2[J] = T
        return T2

    pos = y_true == 1
    neg = y_true == 0

    X1, Y1 = pred1[pos], pred1[neg]
    X2, Y2 = pred2[pos], pred2[neg]

    V10_1 = compute_midrank(np.concatenate([X1, Y1]))[:len(X1)]
    V01_1 = compute_midrank(np.concatenate([Y1, X1]))[:len(Y1)]

    V10_2 = compute_midrank(np.concatenate([X2, Y2]))[:len(X2)]
    V01_2 = compute_midrank(np.concatenate([Y2, X2]))[:len(Y2)]

    S10 = np.cov(V10_1 - V10_2)
    S01 = np.cov(V01_1 - V01_2)

    var = S10 / len(X1) + S01 / len(Y1)
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return p

# ---------- AUC + CI ----------
def compute_auc_ci(df_eval):
    y_train = Surv.from_dataframe("BCR", "BCRmonth", df_train)
    y_eval = Surv.from_dataframe("BCR", "BCRmonth", df_eval)

    risk_c = cph_comb.predict_partial_hazard(df_eval).values
    risk_p = cph_capra.predict_partial_hazard(df_eval).values

    auc_c, _ = cumulative_dynamic_auc(y_train, y_eval, risk_c, times)
    auc_p, _ = cumulative_dynamic_auc(y_train, y_eval, risk_p, times)

    auc_c_boot = []
    auc_p_boot = []

    idx = np.arange(len(df_eval))
    for _ in range(N_BOOT):
        s = np.random.choice(idx, size=len(idx), replace=True)
        d = df_eval.iloc[s]

        if d["BCR"].sum() == 0:
            continue

        y_b = Surv.from_dataframe("BCR", "BCRmonth", d)
        rc = cph_comb.predict_partial_hazard(d).values
        rp = cph_capra.predict_partial_hazard(d).values

        ac, _ = cumulative_dynamic_auc(y_train, y_b, rc, times)
        ap, _ = cumulative_dynamic_auc(y_train, y_b, rp, times)

        auc_c_boot.append(ac)
        auc_p_boot.append(ap)

    auc_c_boot = np.array(auc_c_boot, dtype=float)
    auc_p_boot = np.array(auc_p_boot, dtype=float)

    # Filter bootstrap rows with any NaN AUC to avoid NaN CI at early time points.
    valid_c = ~np.isnan(auc_c_boot).any(axis=1)
    valid_p = ~np.isnan(auc_p_boot).any(axis=1)

    ci_c = np.nanpercentile(auc_c_boot[valid_c], [2.5, 97.5], axis=0)
    ci_p = np.nanpercentile(auc_p_boot[valid_p], [2.5, 97.5], axis=0)

    bad_c = (~valid_c).sum()
    bad_p = (~valid_p).sum()
    if bad_c > 0 or bad_p > 0:
        print(
            f"[{df_eval['Dataset_Type'].iloc[0]}] dropped bootstrap replicates "
            f"with NaN AUC -> Combined: {bad_c}, CAPRA: {bad_p}"
        )

    return auc_c, ci_c, auc_p, ci_p

def safe_yerr(auc, ci, label):
    lower = auc - ci[0]
    upper = ci[1] - auc
    yerr = np.vstack([lower, upper]).astype(float)

    invalid_cols = np.isnan(yerr).any(axis=0)
    if invalid_cols.any():
        bad_times = [f"{times[i]}mo" for i in np.where(invalid_cols)[0]]
        print(f"[Warning] {label} CI unavailable at {', '.join(bad_times)}; error bar skipped.")
        yerr[:, invalid_cols] = 0.0
    return yerr

# ---------- 显著性星号 ----------
def p_to_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"
def add_bar_labels(bars, dy=0.015, fmt="{:.3f}"):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=11
        )

# ---------- 绘图 ----------
for name, data in [
    ("Training", df_train),
    ("Internal_Validation", df_internal),
    ("External_Validation", df_external),
]:
    auc_c, ci_c, auc_p, ci_p = compute_auc_ci(data)

    x = np.arange(len(times))
    width = 0.35

    plt.figure(figsize=(7, 5))
    bars_p = plt.bar(x - width/2, auc_p, width, label="CAPRA",
                     yerr=safe_yerr(auc_p, ci_p, f"{name} CAPRA"),
                     capsize=5)
    bars_c = plt.bar(x + width/2, auc_c, width, label="Combined",
                     yerr=safe_yerr(auc_c, ci_c, f"{name} Combined"),
                     capsize=5)
    add_bar_labels(bars_p)
    add_bar_labels(bars_c)

    plt.xticks(x, [f"{t} mo" for t in times])
    plt.ylim(0.2, 1)
    plt.ylabel("Time-dependent AUC", fontweight="bold")
    plt.title(f"{name}: AUC Comparison", fontweight="bold")
    plt.legend(loc="lower right", prop={"weight": "bold"})

    # 去掉上/右边框，并加粗左/下坐标轴
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)

    plt.tight_layout()
    plt.show()


print(f"\n✅ Results saved to: {OUT_EXCEL}")
