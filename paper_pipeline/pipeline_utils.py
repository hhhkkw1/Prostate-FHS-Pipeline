import json
import os
from typing import Dict, List

import pandas as pd


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
