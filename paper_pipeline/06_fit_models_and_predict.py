import argparse
import os
import warnings
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning, ConvergenceWarning as SMConvergenceWarning

from pipeline_utils import detect_case_id_col, load_config, normalize_case_id, run_publication_analyses


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
    numeric = pd.to_numeric(series, errors='coerce')
    return numeric.fillna(clean.map(t_map))


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

    # Unknown pathology must not silently become a negative AP endpoint.
    parts = out[components + [lymph_col]].apply(pd.to_numeric, errors='coerce')
    if not parts.isin([0, 1]).all().all():
        raise ValueError('Cannot derive AP from missing/nonbinary pathology components; provide adjudicated AP_Status.')
    out[y_col] = (parts.sum(axis=1) > 0).astype(int)
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
    constant_cols = [c for c in x_num if x_num[c].nunique(dropna=True)<2]
    if constant_cols:
        raise ValueError(f'Unestimable association predictors in {model_name}: {constant_cols}')

    x_sm = sm.add_constant(x_num)
    y_int = y.astype(int)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',PerfectSeparationWarning)
            warnings.simplefilter('error',SMConvergenceWarning)
            fit = sm.Logit(y_int, x_sm).fit(disp=False)
        if not fit.mle_retvals.get('converged',False):
            raise RuntimeError(f'Unpenalized association fit did not converge: {model_name}')
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
    except (np.linalg.LinAlgError, PerfectSeparationError, PerfectSeparationWarning, SMConvergenceWarning) as exc:
        raise RuntimeError(f'Unpenalized association fit failed for {model_name}; no automatic change of estimator.') from exc

    return out[out["Variable"] != "const"].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fit logistic models and generate probability predictions.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    parser.add_argument('--analysis-only',action='store_true',help='Evaluate existing configured prediction workbook without refitting.')
    parser.add_argument('--no-plots',action='store_true',help='Export numeric results only.')
    parser.add_argument('--fit-only',action='store_true',help='Fit binary models without downstream publication reports.')
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.analysis_only:
        if args.fit_only:
            parser.error('--analysis-only and --fit-only are mutually exclusive.')
        run_publication_analyses(pd.read_excel(cfg['paths']['prediction_output_excel']),cfg,not args.no_plots)
        return

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
    # Canonical spelling is used by both Python and R; other cohorts remain
    # outside the three development/validation analysis cohorts.
    cohort_lookup = {str(v).strip().lower():v for v in cfg['dataset_values'].values()}
    df[split_col] = df[split_col].astype(str).str.strip().map(lambda x:cohort_lookup.get(x.lower(),x))
    train_mask = df[split_col] == training_value
    train_df = df.loc[train_mask].copy()
    train_df = train_df[train_df[y_col].isin([0, 1])].copy()
    if train_df.empty:
        raise ValueError("No valid training rows with binary outcome.")
    train_df[y_col] = train_df[y_col].astype(int)
    if train_df[y_col].nunique() != 2:
        raise ValueError('Training outcomes need both classes.')

    fill_cols = [age_col, psa_col, pct_pos_col, t_stage_col, gg_bp_col, pirads_col, capra_col, habitat_col]
    medians = train_df[fill_cols].median(numeric_only=True)
    if not np.isfinite(medians).all():
        raise ValueError('A predictor lacks a finite Training median; arbitrary zero filling is not allowed.')
    df[fill_cols] = df[fill_cols].fillna(medians)

    train_df = df.loc[train_mask].copy()
    train_df = train_df[train_df[y_col].isin([0, 1])].copy()
    train_df[y_col] = train_df[y_col].astype(int)
    y_train = train_df[y_col]

    if not np.isfinite(df[fill_cols].to_numpy()).all():
        raise ValueError('Nonfinite model inputs after training-only median imputation.')

    feat_clin = [age_col, psa_col, pct_pos_col, t_stage_col, gg_bp_col]
    feat_image = [pirads_col]
    feat_habitat = [habitat_col]
    feat_clin_pirads = feat_clin + feat_image
    feat_combined = feat_clin + feat_habitat + feat_image
    feat_capra = [capra_col]

    models = {
        "CAPRA": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Clinical": LogisticRegression(max_iter=1000, solver="liblinear"),
        "ImageModel": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Habitat": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Clin_PIRADS": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Combined": LogisticRegression(max_iter=1000, solver="liblinear"),
    }
    feature_map = {
        "CAPRA": feat_capra,
        "Clinical": feat_clin,
        "ImageModel": feat_image,
        "Habitat": feat_habitat,
        "Clin_PIRADS": feat_clin_pirads,
        "Combined": feat_combined,
    }

    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(train_df[feature_map[name]], y_train)

    df["pred_prob_CAPRA"] = models["CAPRA"].predict_proba(df[feat_capra])[:, 1]
    df["pred_prob_Clinical"] = models["Clinical"].predict_proba(df[feat_clin])[:, 1]
    df["pred_prob_ImageModel"] = models["ImageModel"].predict_proba(df[feat_image])[:, 1]
    df["pred_prob_Habitat"] = models["Habitat"].predict_proba(df[feat_habitat])[:, 1]
    df["pred_prob_Clin_PIRADS"] = models["Clin_PIRADS"].predict_proba(df[feat_clin_pirads])[:, 1]
    df["pred_prob_Combined"] = models["Combined"].predict_proba(df[feat_combined])[:, 1]
    for outfile in [pred_outfile,param_outfile]:
        Path(outfile).parent.mkdir(parents=True,exist_ok=True)
    df.to_excel(pred_outfile, index=False)

    params = []
    for name in ["CAPRA", "Clinical", "ImageModel", "Habitat", "Clin_PIRADS", "Combined"]:
        params.append(extract_logistic_params(train_df[feature_map[name]], y_train, name))
    params_df = pd.concat(params, ignore_index=True)

    eval_df = df[df[y_col].isin([0, 1]) & df[split_col].isin(cfg['dataset_values'].values())].copy()
    groups = {
        "All_Analysis_Cohorts": eval_df,
        training_value: eval_df[eval_df[split_col].astype(str).str.strip() == training_value],
        internal_value: eval_df[eval_df[split_col].astype(str).str.strip() == internal_value],
        external_value: eval_df[eval_df[split_col].astype(str).str.strip() == external_value],
    }
    pred_map = {
        "CAPRA": "pred_prob_CAPRA",
        "Clinical": "pred_prob_Clinical",
        "ImageModel": "pred_prob_ImageModel",
        "Habitat": "pred_prob_Habitat",
        "Clin_PIRADS": "pred_prob_Clin_PIRADS",
        "Combined": "pred_prob_Combined",
    }
    metric_rows = []
    for model_name, pred_col in pred_map.items():
        for group_name, group_df in groups.items():
            metric_rows.append(evaluate_predictions(group_df, pred_col, y_col, model_name, group_name))
    metrics_df = pd.DataFrame(metric_rows)

    predictive_rows=[]
    for name,model in models.items():
        for variable,beta in zip(['Intercept',*feature_map[name]],[model.intercept_[0],*model.coef_[0]]):
            predictive_rows.append({'Model':name,'Variable':variable,'Beta':beta,'Fit':'sklearn liblinear L2, C=1',
                                    'N_Training':len(train_df)})
    metadata={'Prediction_fit':'L2-penalized LogisticRegression, liblinear, C=1, max_iter=1000; default tolerance',
              'Association_fit':'Separate unpenalized Training statsmodels Logit; NOT prediction-model coefficients',
              'Ordinal_coding':'T stage, GG and PI-RADS numeric linear trends; no one-hot encoding',
              'Imputation':'Training valid-outcome medians for predictors; endpoints never imputed',
              'Input_SHA256':hashlib.sha256(Path(infile).read_bytes()).hexdigest(),
              'Analysis_cohorts':json.dumps(list(cfg['dataset_values'].values()))}
    with pd.ExcelWriter(param_outfile, engine="openpyxl") as writer:
        pd.DataFrame(predictive_rows).to_excel(writer,sheet_name='Predictive_Coefficients',index=False)
        params_df.to_excel(writer, sheet_name="Training_Association_OR", index=False)
        metrics_df.to_excel(writer, sheet_name="Model_Performance", index=False)
        medians.rename('Training_Median').reset_index().to_excel(writer,sheet_name='Training_Imputation',index=False)
        pd.DataFrame(metadata.items(),columns=['Item','Details']).to_excel(writer,sheet_name='Methods',index=False)

    print("Saved:", pred_outfile)
    print("Saved:", param_outfile)
    print("Pipeline step completed: fit_models_and_predict")
    if not args.fit_only:
        run_publication_analyses(df,cfg,not args.no_plots)


if __name__ == "__main__":
    main()

