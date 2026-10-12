import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_utils import load_config


def calculate_psa_points(psa: float) -> int:
    if pd.isna(psa):
        return 0
    if psa < 6:
        return 0
    if psa < 10:
        return 1
    if psa < 20:
        return 2
    if psa < 30:
        return 3
    return 4


def calculate_gs_points(gs_value) -> int:
    if pd.isna(gs_value):
        return 0
    nums = [int(x) for x in re.findall(r"\d+", str(gs_value))]
    if len(nums) < 2:
        return 0
    primary, secondary = nums[0], nums[1]
    if primary >= 4:
        return 3
    if secondary >= 4:
        return 1
    return 0


def map_t_stage_to_score(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    t_map = {
        "T1": 0, "T1A": 0, "T1B": 0, "T1C": 0,
        "T2A": 1, "T2B": 2, "T2C": 3,
        "T3A": 4, "T3B": 4, "T4": 6,
    }
    clean = series.astype(str).str.strip().str.upper()
    return clean.map(t_map)


def find_first_existing(cols, candidates) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute CAPRA_score and append to LASSO output table.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    infile = cfg["paths"]["lasso_output_excel"]
    outfile = cfg["paths"].get("capra_output_excel", infile.replace(".xlsx", "_with_CAPRA.xlsx"))

    cc = cfg["clinical_columns"]
    age_col = cc["age"]
    psa_col = cc["psa"]
    t_stage_col = cc["t_stage"]
    pct_pos_col = cc["pct_pos"]
    capra_col = cc["capra"]

    capra_cfg = cfg.get("capra_columns", {})
    gs_candidates = capra_cfg.get("gs_candidates", ["GS_BP", "GS_PB"])
    pos_candidates = capra_cfg.get("positive_cores_candidates", ["阳性针数", "Positive_Cores"])
    neg_candidates = capra_cfg.get("negative_cores_candidates", ["阴性针数", "Negative_Cores"])

    print(f"Reading: {infile}")
    df = pd.read_excel(infile)

    required = [age_col, psa_col, t_stage_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for CAPRA: {missing}")

    gs_col = find_first_existing(df.columns, gs_candidates)
    if gs_col is None:
        raise ValueError(f"Missing Gleason score column. Checked candidates: {gs_candidates}")

    # pct_pos priority: use configured pct_pos column if present; otherwise compute from core counts.
    if pct_pos_col not in df.columns:
        pos_col = find_first_existing(df.columns, pos_candidates)
        neg_col = find_first_existing(df.columns, neg_candidates)
        if pos_col is None or neg_col is None:
            raise ValueError(
                f"Missing {pct_pos_col}, and cannot compute it because core columns are missing. "
                f"Checked positive: {pos_candidates}; negative: {neg_candidates}"
            )
        total_cores = pd.to_numeric(df[pos_col], errors="coerce") + pd.to_numeric(df[neg_col], errors="coerce")
        df[pct_pos_col] = pd.to_numeric(df[pos_col], errors="coerce") / total_cores

    # Numeric normalization
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df[psa_col] = pd.to_numeric(df[psa_col], errors="coerce")
    df[pct_pos_col] = pd.to_numeric(df[pct_pos_col], errors="coerce").fillna(0)
    t_score = map_t_stage_to_score(df[t_stage_col]).fillna(0)

    df["score_age"] = (df[age_col] >= 50).astype(int)
    df["score_psa"] = df[psa_col].apply(calculate_psa_points)
    df["score_gs"] = df[gs_col].apply(calculate_gs_points)
    df["score_t"] = (t_score >= 4).astype(int)
    df["score_pct"] = (df[pct_pos_col] >= 0.34).astype(int)

    df[capra_col] = (
        df["score_age"]
        + df["score_psa"]
        + df["score_gs"]
        + df["score_t"]
        + df["score_pct"]
    )

    print("\nCAPRA distribution:")
    print(df[capra_col].value_counts(dropna=False).sort_index())

    df.to_excel(outfile, index=False)
    print(f"\nSaved: {outfile}")
    print("Pipeline step completed: compute_capra_score")


if __name__ == "__main__":
    main()

