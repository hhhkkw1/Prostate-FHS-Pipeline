import json
import os
from typing import Dict, List

import pandas as pd


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Relative paths are relative to the pipeline folder, not the caller's cwd.
    root = Path(config_path).resolve().parent.parent
    cfg['paths'] = {k: str((root / v).resolve()) if not Path(v).is_absolute() else v
                    for k, v in cfg['paths'].items()}
    analysis,survival=cfg.get('analysis',{}),cfg.get('survival',{})
    for settings,key,default in [(analysis,'brier_bootstrap',2000),
                                 (survival,'cindex_bootstrap',3000),(survival,'auc_bootstrap',1000)]:
        if not isinstance(settings.get(key,default),int) or settings.get(key,default)<2:
            raise ValueError(f'{key} must be an integer >= 2.')
    if analysis.get('dpi',600)<=0 or survival.get('min_censored_followup_months',0)<0:
        raise ValueError('dpi must be positive and minimum follow-up must be nonnegative.')
    return cfg


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_case_id_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of case ID columns found: {candidates}")


def normalize_case_id(df: pd.DataFrame, case_id_col: str, out_col: str = "CaseID") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out[case_id_col].astype(str).str.strip()
    return out


# Shared manuscript evaluation (adapted from verified Table3 v3).
from pathlib import Path
import hashlib
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess

# The order and labels below match manuscript Table 3.
MODEL_MAP = {
    "FHS Biomarker": "pred_prob_Habitat",
    "CAPRA Benchmark": "pred_prob_CAPRA",
    "Clinical Model": "pred_prob_Clinical",
    "Clinical-Imaging Model": "pred_prob_Clin_PIRADS",
    "FHS-Integrated Model": "pred_prob_Combined",
}
COHORTS = {"Training": "Training", "Internal_Val": "Internal validation",
           "External_Val": "Cross-site validation"}
REFERENCE = "FHS-Integrated Model"
MODEL_COLORS = {'FHS Biomarker':'#FF7F0E','CAPRA Benchmark':'#1F77B4',
                'Clinical Model':'#2CA02C','Clinical-Imaging Model':'#9467BD',REFERENCE:'black',
                'Treat all':'grey','Treat none':'grey'}


def load_data(path):
    """Fail on invalid analysis data instead of silently changing denominators."""
    raw = pd.read_excel(path)
    required = ["Dataset_Type", "AP_Status", *MODEL_MAP.values()]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    included = raw["Dataset_Type"].isin(COHORTS)
    excluded = raw.loc[~included, "Dataset_Type"].fillna("<missing>").value_counts().to_dict()
    df = raw.loc[included, required].copy()
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not df["AP_Status"].isin([0, 1]).all():
        raise ValueError("AP_Status must be 0/1 for every included patient.")
    for col in MODEL_MAP.values():
        p = df[col].to_numpy(dtype=float)
        if not (np.isfinite(p) & (p >= 0) & (p <= 1)).all():
            raise ValueError(f"Missing, non-finite or out-of-range probability: {col}")
    df["AP_Status"] = df["AP_Status"].astype(int)
    for cohort in COHORTS:
        y = df.loc[df["Dataset_Type"] == cohort, "AP_Status"]
        if min((y == 0).sum(), (y == 1).sum()) < 2:
            raise ValueError(f"Need at least 2 positive and 2 negative patients in {cohort}.")
    return df, excluded


def training_youden_cutoff(y, p):
    """Return the midpoint for an optimal split, not a rounded ROC score.

    Prediction is positive when p >= cutoff. If distinct splits tie, choose
    the highest specificity (first candidate on descending ROC thresholds).
    Keep the full-precision midpoint for validation and plotting.
    """
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    fpr, tpr, scores = roc_curve(y, p, drop_intermediate=False)
    # Include the all-negative split. For an uninformative or reversed model,
    # J=0 can be optimal; do not force at least one positive classification.
    best = np.max(tpr - fpr)
    candidates = np.flatnonzero(np.isclose(tpr - fpr, best, rtol=0, atol=1e-12))
    i = candidates[0]
    upper = np.inf if i == 0 else float(scores[i])
    below = p[p < upper]
    lower = float(np.max(below)) if below.size else np.nan
    cutoff = (lower + upper) / 2 if below.size else upper
    return {"Youden_Cutoff": cutoff, "Training_Youden_J": float(best),
            "Split_Lower_Exclusive": lower, "Split_Upper_Inclusive": upper,
            "Optimal_Splits_Count": len(candidates)}


def delong_auc_covariance(y, probabilities):
    """Original DeLong structural-component covariance; ties receive 0.5.

    probabilities has shape (n_patients, n_models). Scores are oriented so
    higher means AP positive; do not reverse AUC values below 0.5.
    Reference: DeLong et al., Biometrics 1988, doi:10.2307/2531595.
    """
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    positives, negatives = probabilities[y == 1], probabilities[y == 0]
    if min(len(positives), len(negatives)) < 2:
        raise ValueError("DeLong covariance requires >=2 subjects in each class.")
    pos_components, neg_components, aucs = [], [], []
    for k in range(probabilities.shape[1]):
        a, b = positives[:, k, None], negatives[None, :, k]
        comparisons = (a > b).astype(float) + 0.5 * (a == b)
        pos_components.append(comparisons.mean(axis=1))
        neg_components.append(comparisons.mean(axis=0))
        aucs.append(comparisons.mean())
    covariance = (np.atleast_2d(np.cov(pos_components, ddof=1)) / len(positives)
                  + np.atleast_2d(np.cov(neg_components, ddof=1)) / len(negatives))
    return np.asarray(aucs), covariance


def delong_pvalue(aucs, covariance, model_index, reference_index):
    delta = float(aucs[model_index] - aucs[reference_index])
    var = float(covariance[model_index, model_index] + covariance[reference_index, reference_index]
                - 2 * covariance[model_index, reference_index])
    if var < -1e-12:
        raise ValueError("Negative DeLong difference variance.")
    if var <= 0:
        return 1.0 if abs(delta) <= 1e-12 else np.nan
    return float(2 * norm.sf(abs(delta) / np.sqrt(var)))


def calibration_metrics(y, p):
    # Jointly estimated intercept and slope, as in v2; this is NOT the
    # calibration-in-the-large intercept with slope constrained to one.
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    intercept, slope, status = np.nan, np.nan, "failed"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.Logit(y, sm.add_constant(logit, has_constant="add")).fit(disp=False)
        if fit.mle_retvals.get("converged", False) and np.isfinite(fit.params).all():
            intercept, slope = map(float, fit.params)
            status = "ok" if not caught else "warning: " + "; ".join(str(w.message) for w in caught)
        else:
            status = "not converged"
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"
    bins = np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=True)
    ece = sum(np.mean(bins == b) * abs(np.mean(y[bins == b]) - np.mean(p[bins == b]))
              for b in range(10) if np.any(bins == b))
    return {"Brier": float(np.mean((p - y) ** 2)), "ECE": float(ece),
            "Cal_Intercept": intercept, "Cal_Slope": slope, "Calibration_Status": status}


def classification_metrics(y, p, threshold):
    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    ratio = lambda a, b: a / b if b else np.nan
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "SEN": ratio(tp, tp + fn), "SPEC": ratio(tn, tn + fp),
            "PPV": ratio(tp, tp + fp), "NPV": ratio(tn, tn + fn)}


def build_tables(df, threshold_mode="youden", fixed_threshold=0.5, n_boot=2000, seed=123):
    names = list(MODEL_MAP)
    train = df[df["Dataset_Type"] == "Training"]
    thresholds = []
    for name, col in MODEL_MAP.items():
        info = training_youden_cutoff(train["AP_Status"], train[col])
        used = info["Youden_Cutoff"] if threshold_mode == "youden" else fixed_threshold
        thresholds.append({"Model": name, "Probability_Column": col, **info,
                           "Threshold_Used": used, "Threshold_Mode": threshold_mode})
    thresholds = pd.DataFrame(thresholds)
    rows, checks = [], []
    ref = names.index(REFERENCE)
    for code, label in COHORTS.items():
        cohort = df[df["Dataset_Type"] == code]
        y = cohort["AP_Status"].to_numpy()
        probs = cohort[list(MODEL_MAP.values())].to_numpy()
        aucs, covariance = delong_auc_covariance(y, probs)
        for k, name in enumerate(names):
            p = probs[:, k]
            threshold = float(thresholds.loc[k, "Threshold_Used"])
            counts = classification_metrics(y, p, threshold)
            metrics = calibration_metrics(y, p)
            delta, blo, bhi, bp = (0., np.nan, np.nan, np.nan) if k == ref else paired_brier_comparison(y,p,probs[:,ref],n_boot,seed)
            se = np.sqrt(max(0, covariance[k, k]))
            auc = float(aucs[k])
            if not np.isclose(auc, roc_auc_score(y, p), atol=1e-12, rtol=0):
                raise AssertionError("DeLong AUC disagrees with sklearn AUC.")
            row = {"Cohort": label, "Dataset_Type": code, "Model": name,
                   "Probability_Column": MODEL_MAP[name], "N": len(y), "Events": int(y.sum()),
                   "NonEvents": int((y == 0).sum()), "AUC": auc,
                   "AUC_CI_Lower": max(0, auc - norm.ppf(.975) * se),
                   "AUC_CI_Upper": min(1, auc + norm.ppf(.975) * se),
                   "Youden Cutoff": float(thresholds.loc[k, "Youden_Cutoff"]),
                   "Threshold_Used": threshold, "Threshold_Mode": threshold_mode,
                   **counts, "DeLong p": np.nan if k == ref else delong_pvalue(aucs, covariance, k, ref),
                   **metrics, "Brier_p": bp, "Brier_Difference_vs_Ref":delta,
                   "Brier_Diff_CI_Lower":blo,"Brier_Diff_CI_Upper":bhi}
            rows.append(row)
            checks.append({"Cohort": label, "Model": name,
                           "N_equals_sum_CM": len(y) == sum(counts[a] for a in ["TN", "FP", "FN", "TP"]),
                           "Events_equals_FN_TP": int(y.sum()) == counts["FN"] + counts["TP"],
                           "NonEvents_equals_TN_FP": int((y == 0).sum()) == counts["TN"] + counts["FP"],
                           "AUC_matches_sklearn": True})
    raw = pd.DataFrame(rows)
    formatted = raw[["Cohort", "Model"]].copy()
    formatted["AUC (95% CI)"] = raw.apply(
        lambda r: f'{r.AUC:.3f} ({r.AUC_CI_Lower:.3f}–{r.AUC_CI_Upper:.3f})', axis=1)
    cutoff_label = "Youden Cutoff" if threshold_mode == "youden" else "Fixed Cutoff"
    formatted[cutoff_label] = raw["Threshold_Used"].map(lambda x: f"{x:.3f}")
    for col in ["SEN", "SPEC", "PPV", "NPV", "DeLong p", "Brier", "Brier_p", "Cal_Intercept", "Cal_Slope"]:
        if col in ["DeLong p", "Brier_p"]:
            formatted[col] = ["Ref" if name == REFERENCE else format_p(value)
                              for name, value in zip(raw.Model, raw[col])]
        else:
            formatted[col] = raw[col].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "NA")
    return raw, formatted, thresholds, pd.DataFrame(checks)


def format_p(p):
    if not np.isfinite(p):
        return "NA"
    # Avoid printing 0.050* for a value that actually rounds to 0.050.
    text = "<0.001" if p < .001 else (f"{p:.4f}" if p < .05 and round(p, 3) >= .05 else f"{p:.3f}")
    return text + ("*" if p < .05 else "")


def plot_calibration(ax, y, p, model_name, cohort_name):
    observed, predicted = calibration_curve(y, p, n_bins=15, strategy="quantile")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.plot(predicted, observed, "o", color="black", markersize=3.5, alpha=.8, label="Binned observed")
    if len(predicted) >= 4:
        smooth = lowess(observed, predicted, frac=.6, it=0, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color="black", linewidth=2, label="LOWESS")
    else:
        ax.plot(predicted, observed, color="black", linewidth=2, label="Curve")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed probability")
    ax.set_title(f"{model_name}\n{cohort_name}", fontsize=8, weight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def plot_confusion(ax, row):
    # Draw the exported counts directly: no second threshold calculation.
    cm = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
    sums = cm.sum(axis=1, keepdims=True)
    colors = np.divide(cm, sums, out=np.zeros_like(cm, dtype=float), where=sums != 0)
    sns.heatmap(colors, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
                cbar=False, square=True, linewidths=.8, ax=ax, annot_kws={"size": 10})
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_title(f'{row.Model}\n{row.Cohort}; cutoff={row.Threshold_Used:.6f}', fontsize=8, weight="bold")
    ax.set_xticklabels(["AP-", "AP+"], fontsize=8)
    ax.set_yticklabels(["AP-", "AP+"], rotation=90, va="center", fontsize=8)


def make_plots(df, raw, output_dir, dpi):
    for name, col in MODEL_MAP.items():
        fig_cal, axes_cal = plt.subplots(1, 3, figsize=(10, 3.3))
        fig_cm, axes_cm = plt.subplots(1, 3, figsize=(10, 3.3))
        for i, (code, label) in enumerate(COHORTS.items()):
            cohort = df[df["Dataset_Type"] == code]
            row = raw[(raw.Dataset_Type == code) & (raw.Model == name)].iloc[0]
            plot_calibration(axes_cal[i], cohort.AP_Status.to_numpy(), cohort[col].to_numpy(), name, label)
            plot_confusion(axes_cm[i], row)
        safe = name.replace(" ", "_").replace("-", "_")
        for kind, fig in [("Calibration", fig_cal), ("ConfusionMatrix", fig_cm)]:
            fig.tight_layout()
            fig.savefig(output_dir / f"{kind}_{safe}_v3.tiff", dpi=dpi,
                        pil_kwargs={"compression": "tiff_lzw"})
            fig.savefig(output_dir / f"{kind}_{safe}_v3.png", dpi=180)
            plt.close(fig)


def paired_brier_comparison(y, p, reference, n_boot=2000, seed=123):
    """Paired patient bootstrap, percentile CI and normal bootstrap-SE test.

    Same method as the updated R evaluation script. The test is an approximate
    two-sided normal test, not a bootstrap tail probability or paired t test.
    Predictions are held fixed; training estimates are apparent performance.
    """
    d = (np.asarray(p) - y)**2 - (np.asarray(reference) - y)**2
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(n_boot)])
    se = means.std(ddof=1)
    pvalue = float(2 * norm.sf(abs(d.mean()) / se)) if se > 0 else (1.0 if np.all(d == 0) else np.nan)
    lo, hi = np.quantile(means, [.025, .975])
    return float(d.mean()), float(lo), float(hi), pvalue


def canonical_analysis_data(raw, cfg):
    """Canonical cohort/endpoint columns used by all binary reporting functions."""
    df = raw.copy()
    split, outcome = cfg['columns']['split_col'], cfg['columns']['outcome_col']
    if 'pred_prob_ImageModel' not in df and 'pred_prob_Image' in df:
        df['pred_prob_ImageModel'] = df['pred_prob_Image']
    ds = cfg['dataset_values']
    values = {str(ds[k]).strip().lower(): v for k, v in
              [('training', 'Training'), ('internal_val', 'Internal_Val'), ('external_val', 'External_Val')]}
    df['Dataset_Type'] = df[split].astype(str).str.strip().str.lower().map(values)
    excluded = raw.loc[df['Dataset_Type'].isna(), split].fillna('<missing>').value_counts().to_dict()
    df = df.loc[df['Dataset_Type'].notna()].copy()
    df['AP_Status'] = pd.to_numeric(df[outcome], errors='coerce')
    if not df['AP_Status'].isin([0, 1]).all():
        raise ValueError('Every included binary-analysis patient must have a valid 0/1 outcome.')
    for c in [*MODEL_MAP.values(), 'pred_prob_ImageModel']:
        p = pd.to_numeric(df[c], errors='coerce').to_numpy(float)
        if not (np.isfinite(p) & (p >= 0) & (p <= 1)).all():
            raise ValueError(f'Invalid or missing probability in {c}; paired analyses need the same patients.')
        df[c] = p
    for code in COHORTS:
        y = df.loc[df.Dataset_Type == code, 'AP_Status']
        if min((y == 0).sum(), (y == 1).sum()) < 2:
            raise ValueError(f'{code} needs at least two events and non-events.')
    return df, excluded


def net_benefit(y, p, thresholds):
    y, p, thresholds = np.asarray(y), np.asarray(p), np.asarray(thresholds)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p) or len(y)==0:
        raise ValueError('DCA requires nonempty paired one-dimensional outcomes and probabilities.')
    if not np.isin(y,[0,1]).all() or not (np.isfinite(p)&(p>=0)&(p<=1)).all():
        raise ValueError('DCA requires binary outcomes and valid probabilities in [0,1].')
    if not np.all((thresholds > 0) & (thresholds < 1)):
        raise ValueError('DCA thresholds must lie strictly between 0 and 1.')
    positive = p[:, None] >= thresholds[None, :]
    tp = (positive & (y[:, None] == 1)).sum(axis=0)
    fp = (positive & (y[:, None] == 0)).sum(axis=0)
    return tp / len(y) - fp / len(y) * thresholds / (1 - thresholds)


def export_binary_reports(raw_df, cfg, make_figures=True):
    output = Path(cfg['paths']['plot_dir'])
    output.mkdir(parents=True, exist_ok=True)
    df, excluded = canonical_analysis_data(raw_df, cfg)
    settings = cfg.get('analysis', {})
    seed, nboot = settings.get('seed', 123), settings.get('brier_bootstrap', 2000)
    raw, formatted, thresholds, checks = build_tables(df,n_boot=nboot,seed=seed)
    methods = {
        'Included_N': len(df), 'Excluded_cohorts': json.dumps(excluded, ensure_ascii=False),
        'Threshold': 'Training Youden midpoint; ties choose highest specificity; full precision frozen in validation',
        'AUC': 'Higher probabilities indicate AP; paired DeLong test, normal 95% CI clipped to [0,1]',
        'Brier_p': f'Two-sided normal test using SE of {nboot} paired patient bootstrap replicates; seed={seed}',
        'Brier_CI': 'Percentile 95% CI of Brier(model)-Brier(integrated); positive favors integrated',
        'Calibration': 'Joint intercept/slope from logistic regression on logit(p), clipped at 1e-6',
        'Inference': 'Conditional on fixed fitted predictions; training is apparent; unadjusted comparisons',
        'Plot': '15 quantile calibration bins, LOWESS on bin points; confusion plots use exported counts',
    }
    tables = [('Table3_Formatted', formatted), ('Table3_Raw', raw), ('Training_Thresholds', thresholds),
              ('Consistency_Checks', checks), ('Methods', pd.DataFrame(methods.items(), columns=['Item','Details']))]
    if not checks.iloc[:, 2:].to_numpy().all():
        raise AssertionError('Table3 consistency checks failed.')
    with pd.ExcelWriter(output / 'Table3_5Models_v3.xlsx') as writer:
        for sheet, table in tables:
            table.to_excel(writer, sheet_name=sheet, index=False)
    raw.to_csv(output / 'Table3_Raw_v3.csv', index=False)
    thresholds.to_csv(output / 'Training_Thresholds.csv', index=False)
    # Six-model raw evaluation drives step 7's ROC workbooks. Five manuscript
    # models share the exact same rows and bootstrap P values as Table3.
    all_metrics = raw.copy()
    for code, label in COHORTS.items():
        sub = df[df.Dataset_Type == code]
        y, p = sub.AP_Status.to_numpy(), sub.pred_prob_ImageModel.to_numpy()
        refp = sub[MODEL_MAP[REFERENCE]].to_numpy()
        aucs, cov = delong_auc_covariance(y, np.column_stack([p, refp]))
        train = df[df.Dataset_Type == 'Training']
        cutoff = training_youden_cutoff(train.AP_Status, train.pred_prob_ImageModel)['Youden_Cutoff']
        delta, lo, hi, pv = paired_brier_comparison(y, p, refp, nboot, seed)
        se = np.sqrt(cov[0, 0])
        row = {'Dataset_Type': code, 'Cohort': label, 'Model': 'Image Model',
               'Probability_Column': 'pred_prob_ImageModel', 'N': len(y), 'Events': int(y.sum()),
               'NonEvents': int((y == 0).sum()), 'AUC': aucs[0],
               'AUC_CI_Lower': max(0, aucs[0]-norm.ppf(.975)*se),
               'AUC_CI_Upper': min(1, aucs[0]+norm.ppf(.975)*se), 'Youden Cutoff': cutoff,
               'Threshold_Used': cutoff, 'Threshold_Mode': 'youden', **classification_metrics(y,p,cutoff),
               **calibration_metrics(y,p), 'DeLong p': delong_pvalue(aucs,cov,0,1),
               'Brier_p': pv, 'Brier_Difference_vs_Ref': delta, 'Brier_Diff_CI_Lower':lo, 'Brier_Diff_CI_Upper':hi}
        all_metrics = pd.concat([all_metrics, pd.DataFrame([row])], ignore_index=True)
    all_metrics.to_csv(output / 'Binary_Model_Metrics.csv', index=False)
    prediction_file=Path(cfg['paths']['prediction_output_excel'])
    provenance={'prediction_md5':hashlib.md5(prediction_file.read_bytes()).hexdigest(),
                'prediction_sha256':hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
                'included_n':len(df),'excluded_cohorts':excluded,'brier_bootstrap':nboot,'seed':seed}
    (output/'Binary_Report_Provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    if make_figures:
        make_plots(df, raw, output, settings.get('dpi',600))
    dca_rows = []
    thresholds_dca = np.linspace(.01,.99,120)
    for code, label in COHORTS.items():
        sub = df[df.Dataset_Type == code]
        y = sub.AP_Status.to_numpy()
        curves = {m: net_benefit(y,sub[c].to_numpy(),thresholds_dca) for m,c in MODEL_MAP.items()}
        curves['Treat all'] = y.mean()-(1-y.mean())*thresholds_dca/(1-thresholds_dca)
        curves['Treat none'] = np.zeros_like(thresholds_dca)
        for model, nb in curves.items():
            dca_rows.extend({'Dataset_Type':code,'Model':model,'N':len(y),'Threshold':t,'Net_Benefit':v}
                            for t,v in zip(thresholds_dca,nb))
        if make_figures:
            fig, ax = plt.subplots(figsize=(5,5))
            for model, nb in curves.items():
                ax.plot(thresholds_dca,nb,label=model,linewidth=2 if model==REFERENCE else 1.2,
                        color=MODEL_COLORS[model],linestyle='--' if model in ['Treat all','Clinical Model']
                        else ':' if model in ['Treat none','FHS Biomarker'] else '-.' if model=='Clinical-Imaging Model' else '-')
            # Permit negative model benefit; raw CSV retains the full treat-all
            # tail, which tends to minus infinity as the threshold approaches 1.
            lower = min(-.1, min(float(v.min()) for m,v in curves.items() if m not in ['Treat all','Treat none'])-.03)
            ax.set(xlim=(.01,.99),ylim=(lower,float(y.mean())+.08),xlabel='Threshold probability',
                   ylabel='Net benefit',title=f'Decision curve analysis\n{label}')
            ax.legend(fontsize=7,frameon=False)
            fig.tight_layout()
            fig.savefig(output/f'DCA_{code}.tiff',dpi=settings.get('dpi',600),pil_kwargs={'compression':'tiff_lzw'})
            plt.close(fig)
    pd.DataFrame(dca_rows).to_csv(output/'DCA_Net_Benefit.csv',index=False)
    return df


def export_cox_reports(raw, cfg, make_figures=True):
    """Training-only Cox fits; paired conditional bootstrap C and IPCW AUC CIs.

    Missing endpoints are excluded, never imputed. No ordinary DeLong test is
    applied to censored outcomes. The old bootstrap fraction below zero is not
    labelled a P value; report paired delta C with a percentile interval instead.
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from lifelines.statistics import proportional_hazard_test
    from sksurv.util import Surv
    from sksurv.metrics import cumulative_dynamic_auc

    settings = cfg.get('survival', {})
    if not settings.get('enabled', True):
        return
    output = Path(cfg['paths']['plot_dir'])
    output.mkdir(parents=True, exist_ok=True)
    split = cfg['columns']['split_col']
    event, time = settings.get('event_col','BCR'),settings.get('time_col','BCRmonth')
    if event not in raw or time not in raw:
        raise ValueError('Survival analysis enabled but BCR/time columns absent; explicitly disable it in config if unavailable.')
    cc = cfg['clinical_columns']
    clinical = [cc[k] for k in ['age','psa','pct_pos','t_stage','gg_bp','pirads']]
    fhs, capra = cc['habitat_score'],cc['capra']
    covariates = clinical + [fhs,capra]
    expected = list(cfg['dataset_values'].values())
    df = raw.loc[raw[split].isin(expected)].copy()
    missing_before = df[[event,time]].isna().any(axis=1)
    for c in [event,time,*covariates]:
        df[c] = pd.to_numeric(df[c],errors='coerce')
    invalid_event = df[event].notna() & ~df[event].isin([0,1])
    if invalid_event.any():
        raise ValueError('Invalid BCR event coding; expected 0/1 or missing.')
    # A nonnumeric nonmissing endpoint is a data error, not a missing value.
    invalid_numeric = ~missing_before & (~np.isfinite(df[event]) | ~np.isfinite(df[time]))
    if invalid_numeric.any() or (df[time].dropna() <= 0).any():
        raise ValueError('BCR and follow-up times must be finite; times must be positive months.')
    missing_n = int(df[[event,time]].isna().any(axis=1).sum())
    df = df.dropna(subset=[event,time]).copy()
    minimum = float(settings.get('min_censored_followup_months',0))
    short = (df[event]==0) & (df[time]<minimum)
    excluded_short = int(short.sum())
    df = df.loc[~short].copy()
    training = cfg['dataset_values']['training']
    train = df.loc[df[split]==training].copy()
    medians = train[covariates].median()
    if not np.isfinite(medians).all():
        raise ValueError('A Cox predictor has no finite training median.')
    df[covariates] = df[covariates].fillna(medians)
    if not np.isfinite(df[covariates].to_numpy()).all():
        raise ValueError('Nonfinite Cox predictors after training-only imputation.')
    train = df.loc[df[split]==training].copy()
    combined_vars = [fhs,cc['t_stage'],cc['psa'],cc['gg_bp'],cc['age'],cc['pct_pos'],cc['pirads']]
    combined = CoxPHFitter().fit(train[[time,event,*combined_vars]],duration_col=time,event_col=event)
    baseline = CoxPHFitter().fit(train[[time,event,capra]],duration_col=time,event_col=event)

    def summary_table(model, kind):
        table = model.summary.reset_index().rename(columns={'covariate':'Variable'})
        table.insert(0,'Analysis',kind)
        table['N'],table['Events'] = len(train),int(train[event].sum())
        return table

    uni = []
    for c in [*clinical,fhs,capra]:
        fitted = CoxPHFitter().fit(train[[time,event,c]],duration_col=time,event_col=event)
        uni.append(summary_table(fitted,'Univariable'))
    uni = pd.concat(uni,ignore_index=True)
    multi = summary_table(combined,'Multivariable')
    capra_summary = summary_table(baseline,'CAPRA baseline')
    ph = []
    for transform in ['rank','km']:
        table = proportional_hazard_test(combined,train[[time,event,*combined_vars]],time_transform=transform).summary.reset_index()
        table['Time_transform'] = transform
        ph.append(table)
    ph = pd.concat(ph,ignore_index=True)
    y_train = Surv.from_arrays(train[event].astype(bool).to_numpy(),train[time].to_numpy(float))
    seed = int(cfg.get('analysis',{}).get('seed',123))
    n_c = int(settings.get('cindex_bootstrap',3000))
    n_auc = int(settings.get('auc_bootstrap',1000))
    times = np.asarray(settings.get('times_months',[12,24,36,48,60]),float)
    c_rows,auc_rows,counts = [],[],[]
    for cohort in expected:
        sub = df.loc[df[split]==cohort].copy()
        if sub.empty:
            raise ValueError(f'No eligible survival patients in {cohort}.')
        e,t = sub[event].astype(bool).to_numpy(),sub[time].to_numpy(float)
        risk = np.column_stack([baseline.predict_log_partial_hazard(sub).to_numpy(),
                                combined.predict_log_partial_hazard(sub).to_numpy()])
        counts.append({'Cohort':cohort,'N':len(sub),'Events':int(e.sum()),
                       'Censored_under_12_months':int(((~e)&(t<12)).sum())})
        point = np.array([concordance_index(t,-risk[:,k],e) for k in [0,1]])
        rng = np.random.default_rng(seed)
        c_boot=[]
        for _ in range(n_c):
            idx = rng.integers(len(sub),size=len(sub))
            try:
                c_boot.append([concordance_index(t[idx],-risk[idx,k],e[idx]) for k in [0,1]])
            except ZeroDivisionError:
                continue
        c_boot=np.asarray(c_boot).reshape(-1,2)
        if len(c_boot)<2:
            raise ValueError(f'No usable concordance bootstrap in {cohort}.')
        delta_boot=c_boot[:,1]-c_boot[:,0]
        dlo,dhi=np.quantile(delta_boot,[.025,.975])
        for k,model in enumerate(['CAPRA Benchmark',REFERENCE]):
            lo,hi=np.quantile(c_boot[:,k],[.025,.975])
            c_rows.append({'Cohort':cohort,'Model':model,'N':len(sub),'Events':int(e.sum()),
                           'C_index':point[k],'CI_Lower':lo,'CI_Upper':hi,
                           'Delta_Integrated_minus_CAPRA':point[1]-point[0],
                           'Delta_CI_Lower':dlo,'Delta_CI_Upper':dhi,'Valid_Bootstrap':len(c_boot),'Requested_Bootstrap':n_c})
        y = Surv.from_arrays(e,t)
        # Bootstrap both models on the SAME sampled patients. Eligibility and
        # censoring support are checked per time, avoiding an invalid 60-month
        # replicate discarding a valid 12-month result.
        auc_point=np.full((len(times),2),np.nan)
        failures=['']*len(times)
        for j,tau in enumerate(times):
            try:
                for k in [0,1]:
                    auc_point[j,k]=cumulative_dynamic_auc(y_train,y,risk[:,k],[tau])[0][0]
            except ValueError as exc:
                auc_point[j,:]=np.nan
                failures[j]=str(exc)
        bootstrap=[[] for _ in times]
        rng=np.random.default_rng(seed)
        for _ in range(n_auc):
            idx=rng.integers(len(sub),size=len(sub))
            for j,tau in enumerate(times):
                if failures[j]:
                    continue
                try:
                    vals=[cumulative_dynamic_auc(y_train,y[idx],risk[idx,k],[tau])[0][0] for k in [0,1]]
                    if np.isfinite(vals).all():
                        bootstrap[j].append(vals)
                except ValueError:
                    continue
        for j,tau in enumerate(times):
            pairs=np.asarray(bootstrap[j]).reshape(-1,2)
            intervals=np.quantile(pairs,[.025,.975],axis=0) if len(pairs)>1 else np.full((2,2),np.nan)
            delta_ci=np.quantile(pairs[:,1]-pairs[:,0],[.025,.975]) if len(pairs)>1 else [np.nan,np.nan]
            for k,model in enumerate(['CAPRA Benchmark',REFERENCE]):
                auc_rows.append({'Cohort':cohort,'Model':model,'Time_months':tau,
                                 'AUC':auc_point[j,k],'CI_Lower':intervals[0,k],'CI_Upper':intervals[1,k],
                                 'Delta_Integrated_minus_CAPRA':auc_point[j,1]-auc_point[j,0],
                                 'Delta_CI_Lower':delta_ci[0],'Delta_CI_Upper':delta_ci[1],
                                 'Cases_by_time':int((e&(t<=tau)).sum()),'Controls_after_time':int((t>tau).sum()),
                                 'Valid_Bootstrap':len(pairs),'Requested_Bootstrap':n_auc,
                                 'Status':failures[j] or ('ok' if len(pairs)>1 else 'CI unavailable')})
    cdf,adf=pd.DataFrame(c_rows),pd.DataFrame(auc_rows)
    methods={'Cohort':'Model fits and medians from eligible Training patients only; validation coefficients frozen',
             'Missing_endpoints_excluded':missing_n,'Min_censored_followup_months':minimum,
             'Short_censored_excluded':excluded_short,'Seed':seed,
             'C_index':'Harrell concordance; minus log partial hazard; paired percentile CI for delta, no tail-fraction P',
             'Time_AUC':'Cumulative/dynamic IPCW; censoring distribution estimated in Training; paired percentile CI',
             'Bootstrap':'Evaluation patients resampled; Cox models NOT refitted; conditional uncertainty only',
             'Variables':'T-stage, GG and PI-RADS numeric ordinal trend; pct_pos per fraction (0 to 1); FHS per raw unit',
             'PH_test':'Rank and KM transforms; non-significance does not prove PH or linearity',
             'Multiplicity':'Exploratory, unadjusted intervals at multiple times'}
    with pd.ExcelWriter(output/'Cox_Results.xlsx') as writer:
        for sheet,table in [('Univariable',uni),('Multivariable',multi),('CAPRA_Model',capra_summary),
                            ('PH_Diagnostics',ph),('C_Index',cdf),('Time_Dependent_AUC',adf),
                            ('Cohort_Counts',pd.DataFrame(counts)),('Training_Imputation',medians.rename('Median').reset_index()),
                            ('Methods',pd.DataFrame(methods.items(),columns=['Item','Details']))]:
            table.to_excel(writer,sheet_name=sheet,index=False)
    cdf.to_csv(output/'Cox_C_Index.csv',index=False)
    adf.to_csv(output/'Cox_Time_Dependent_AUC.csv',index=False)
    if make_figures:
        make_cox_plots(adf,expected,output,cfg.get('analysis',{}).get('dpi',600))


def make_cox_plots(adf,cohorts,output,dpi):
    # Retain the original grouped-bar presentation, but draw actual CI limits
    # directly so an unavailable CI is not fabricated as a zero-length errorbar.
    for cohort in cohorts:
        fig,ax=plt.subplots(figsize=(6,4))
        for k,model in enumerate(['CAPRA Benchmark',REFERENCE]):
            sub=adf[(adf.Cohort==cohort)&(adf.Model==model)]
            x=sub.Time_months.to_numpy()/12+(k-.5)*.36
            color=['#1F77B4','black'][k]
            ax.bar(x,sub.AUC.to_numpy(),width=.32,color=color,label=model,alpha=.8)
            good=np.isfinite(sub.CI_Lower)&np.isfinite(sub.CI_Upper)
            ax.vlines(x[good],sub.loc[good,'CI_Lower'],sub.loc[good,'CI_Upper'],color='black')
            for bound in ['CI_Lower','CI_Upper']:
                ax.hlines(sub.loc[good,bound],x[good]-.05,x[good]+.05,color='black')
        ax.set(xlabel='Years after surgery',ylabel='Time-dependent AUC',ylim=(0,1),title=COHORTS.get(cohort,cohort))
        ax.legend(frameon=False,fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(output)/f'Cox_Time_AUC_{cohort}.tiff',dpi=dpi,pil_kwargs={'compression':'tiff_lzw'})
        plt.close(fig)


def run_publication_analyses(df, cfg, make_figures=True):
    export_binary_reports(df,cfg,make_figures)
    export_cox_reports(df,cfg,make_figures)
    print('Saved Table3, calibration/confusion figures, DCA and Cox reports to',cfg['paths']['plot_dir'])
