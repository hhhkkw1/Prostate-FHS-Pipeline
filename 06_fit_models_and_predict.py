import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from pipeline_utils import detect_case_id_col, load_config, normalize_case_id


def canonical_case_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def map_t_stage_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    t_map = {
        "T1": 0, "T1A": 0, "T1B": 0, "T1C": 0,
        "T2A": 1, "T2B": 2, "T2C": 3,
        "T3A": 4, "T3B": 4, "T4": 6,
    }
    clean = series.astype(str).str.strip().str.upper()
    return clean.map(t_map)


def build_outcome_if_missing(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    y_col = cfg["columns"]["outcome_col"]
    if y_col in out.columns:
        return out

    components = cfg["outcome_components"]
    lymph_col = cfg["lymph_col"]
    missing = [c for c in components + [lymph_col] if c not in out.columns]
    if missing:
        raise ValueError(f"Cannot build {y_col}; missing columns: {missing}")

    out["Lymph_Calc"] = np.where(out[lymph_col].isna(), 0, out[lymph_col])
    out[y_col] = (out[components + ["Lymph_Calc"]].sum(axis=1, skipna=True) > 0).astype(int)
    return out


def calibration_intercept_slope(y_true: np.ndarray, p_pred: np.ndarray):
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    lp = np.log(p / (1 - p))
    x = sm.add_constant(lp)
    try:
        fit = sm.Logit(y, x).fit(disp=False)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def evaluate_predictions(eval_df: pd.DataFrame, pred_col: str, y_col: str, model_name: str, group_name: str) -> Dict:
    sub = eval_df[[y_col, pred_col]].dropna().copy()
    sub = sub[sub[y_col].isin([0, 1])]

    if len(sub) == 0:
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

    y = sub[y_col].astype(int).to_numpy()
    p = sub[pred_col].astype(float).to_numpy()

    if np.unique(y).size < 2:
        auc = np.nan
    else:
        auc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    cal_intercept, cal_slope = calibration_intercept_slope(y, p)

    return {
        "Model": model_name,
        "Dataset_Group": group_name,
        "N": int(len(sub)),
        "Events": int(y.sum()),
        "AUC": auc,
        "Brier": brier,
        "Cal_Intercept": cal_intercept,
        "Cal_Slope": cal_slope,
    }


def extract_logistic_params(x: pd.DataFrame, y: pd.Series, model_name: str) -> pd.DataFrame:
    x_num = x.astype(float).copy()
    # Drop zero-variance columns to reduce singular matrix risk.
    keep_cols = [c for c in x_num.columns if x_num[c].nunique(dropna=True) > 1]
    x_num = x_num[keep_cols]
    if x_num.shape[1] == 0:
        return pd.DataFrame(
            columns=["Model", "Variable", "Beta", "OR", "CI_lower", "CI_upper", "p_value"]
        )

    x_sm = sm.add_constant(x_num)
    y_int = y.astype(int)

    try:
        fit = sm.Logit(y_int, x_sm).fit(disp=False)
        params = fit.params
        conf = fit.conf_int()
        pvals = fit.pvalues

        out = pd.DataFrame(
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
    except (np.linalg.LinAlgError, PerfectSeparationError):
        # Fallback: regularized fit to avoid crash when matrix is singular / separated.
        fit_reg = sm.Logit(y_int, x_sm).fit_regularized(disp=False, alpha=1e-6, maxiter=500)
        params = fit_reg.params
        out = pd.DataFrame(
            {
                "Model": model_name,
                "Variable": params.index,
                "Beta": params.values,
                "OR": np.exp(params.values),
                "CI_lower": np.nan,
                "CI_upper": np.nan,
                "p_value": np.nan,
            }
        )

    return out[out["Variable"] != "const"].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fit logistic models and generate probability predictions.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    infile = cfg["paths"].get("capra_output_excel", cfg["paths"]["lasso_output_excel"])
    if (not os.path.exists(infile)) and ("lasso_output_excel" in cfg["paths"]):
        infile = cfg["paths"]["lasso_output_excel"]
    pred_outfile = cfg["paths"]["prediction_output_excel"]
    param_outfile = cfg["paths"]["model_param_output_excel"]

    split_col = cfg["columns"]["split_col"]
    y_col = cfg["columns"]["outcome_col"]
    training_value = cfg["dataset_values"]["training"]
    internal_value = cfg["dataset_values"]["internal_val"]
    external_value = cfg["dataset_values"]["external_val"]
    cc = cfg["clinical_columns"]

    age_col = cc["age"]
    psa_col = cc["psa"]
    t_stage_col = cc["t_stage"]
    pirads_col = cc["pirads"]
    pct_pos_col = cc["pct_pos"]
    gg_bp_col = cc["gg_bp"]
    capra_col = cc["capra"]
    habitat_col = cc["habitat_score"]

    df = pd.read_excel(infile)
    case_id_col = detect_case_id_col(df, cfg["columns"]["case_id_candidates"])
    df = normalize_case_id(df, case_id_col, out_col="CaseID")
    df["CaseID"] = df["CaseID"].map(canonical_case_id)
    df = build_outcome_if_missing(df, cfg)

    required_cols = [
        split_col,
        y_col,
        age_col,
        psa_col,
        pct_pos_col,
        t_stage_col,
        gg_bp_col,
        pirads_col,
        capra_col,
        habitat_col,
    ]
    if habitat_col not in df.columns and "Rad_score_1se" in df.columns:
        df[habitat_col] = df["Rad_score_1se"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_t_stage = df[t_stage_col].copy()

    numeric_cols = [
        age_col,
        psa_col,
        pct_pos_col,
        gg_bp_col,
        pirads_col,
        capra_col,
        habitat_col,
        y_col,
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # T stage may be text labels (e.g., T2a/T3b); map before numeric coercion.
    df[t_stage_col] = map_t_stage_to_numeric(raw_t_stage)
    unmapped_t = raw_t_stage[df[t_stage_col].isna() & raw_t_stage.notna()].astype(str).unique()
    if len(unmapped_t) > 0:
        print(f"Warning: unmapped T_stage values will be imputed: {sorted(unmapped_t)}")
    train_mask = df[split_col].astype(str).str.strip() == training_value
    train_df = df.loc[train_mask].copy()
    train_df = train_df[train_df[y_col].isin([0, 1])].copy()
    if train_df.empty:
        raise ValueError("No valid training rows with binary outcome.")
    train_df[y_col] = train_df[y_col].astype(int)

    fill_cols = [age_col, psa_col, pct_pos_col, t_stage_col, gg_bp_col, pirads_col, capra_col, habitat_col]
    medians = train_df[fill_cols].median(numeric_only=True)
    for c in fill_cols:
        if c not in medians.index or not np.isfinite(medians[c]):
            medians[c] = 0.0
    df[fill_cols] = df[fill_cols].fillna(medians)

    train_df = df.loc[train_mask].copy()
    train_df = train_df[train_df[y_col].isin([0, 1])].copy()
    train_df[y_col] = train_df[y_col].astype(int)
    y_train = train_df[y_col]

    # Final NaN guard for model inputs.
    if df[fill_cols].isna().any().any():
        df[fill_cols] = df[fill_cols].fillna(0.0)
    if train_df[fill_cols].isna().any().any():
        train_df[fill_cols] = train_df[fill_cols].fillna(0.0)

    feat_clin = [age_col, psa_col, pct_pos_col, t_stage_col, gg_bp_col]
    feat_image = [pirads_col]
    feat_habitat = [habitat_col]
    feat_combined = feat_clin + feat_image + feat_habitat
    feat_capra = [capra_col]

    models = {
        "Clinical": LogisticRegression(max_iter=1000, solver="liblinear"),
        "ImageModel": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Habitat": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Combined": LogisticRegression(max_iter=1000, solver="liblinear"),
        "CAPRA": LogisticRegression(max_iter=1000, solver="liblinear"),
    }
    feature_map = {
        "Clinical": feat_clin,
        "ImageModel": feat_image,
        "Habitat": feat_habitat,
        "Combined": feat_combined,
        "CAPRA": feat_capra,
    }

    for name, model in models.items():
        model.fit(train_df[feature_map[name]], y_train)

    df["pred_prob_Clinical"] = models["Clinical"].predict_proba(df[feat_clin])[:, 1]
    df["pred_prob_ImageModel"] = models["ImageModel"].predict_proba(df[feat_image])[:, 1]
    df["pred_prob_Habitat"] = models["Habitat"].predict_proba(df[feat_habitat])[:, 1]
    df["pred_prob_Combined"] = models["Combined"].predict_proba(df[feat_combined])[:, 1]
    df["pred_prob_CAPRA"] = models["CAPRA"].predict_proba(df[feat_capra])[:, 1]
    df.to_excel(pred_outfile, index=False)

    params = []
    for name in ["Clinical", "ImageModel", "Habitat", "Combined", "CAPRA"]:
        params.append(extract_logistic_params(train_df[feature_map[name]], y_train, name))
    params_df = pd.concat(params, ignore_index=True)

    eval_df = df[df[y_col].isin([0, 1])].copy()
    groups = {
        "All_Valid": eval_df,
        training_value: eval_df[eval_df[split_col].astype(str).str.strip() == training_value],
        internal_value: eval_df[eval_df[split_col].astype(str).str.strip() == internal_value],
        external_value: eval_df[eval_df[split_col].astype(str).str.strip() == external_value],
    }
    pred_map = {
        "Clinical": "pred_prob_Clinical",
        "ImageModel": "pred_prob_ImageModel",
        "Habitat": "pred_prob_Habitat",
        "Combined": "pred_prob_Combined",
        "CAPRA": "pred_prob_CAPRA",
    }
    metric_rows = []
    for model_name, pred_col in pred_map.items():
        for group_name, group_df in groups.items():
            metric_rows.append(evaluate_predictions(group_df, pred_col, y_col, model_name, group_name))
    metrics_df = pd.DataFrame(metric_rows)

    with pd.ExcelWriter(param_outfile, engine="openpyxl") as writer:
        params_df.to_excel(writer, sheet_name="Logistic_Params", index=False)
        metrics_df.to_excel(writer, sheet_name="Model_Performance", index=False)

    print("Saved:", pred_outfile)
    print("Saved:", param_outfile)
    print("Pipeline step completed: fit_models_and_predict")


if __name__ == "__main__":
    main()

