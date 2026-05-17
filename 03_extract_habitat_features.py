import argparse
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import center_of_mass, distance_transform_edt, label
from skimage.measure import marching_cubes, mesh_surface_area
from tqdm import tqdm

from pipeline_utils import detect_case_id_col, load_config, normalize_case_id


def canonical_case_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_label_map(num_habitats: int) -> List[Tuple[int, int]]:
    return [(i + 1, i) for i in range(num_habitats)]


def load_case_arrays(d_path: str, f_path: str, roi_path: str):
    roi_img = nib.load(roi_path)
    roi = roi_img.get_fdata().astype(int)
    voxel_sizes = roi_img.header.get_zooms()[:3]
    voxel_volume = float(np.prod(voxel_sizes))

    d_img = nib.load(d_path).get_fdata()
    f_img = nib.load(f_path).get_fdata()
    return roi, d_img, f_img, voxel_sizes, voxel_volume


def empty_feature_row(prefix: str) -> Dict[str, float]:
    keys = [
        "Vol",
        "Ratio",
        "Sphericity",
        "d_Mean",
        "d_Std",
        "d_P10",
        "d_P90",
        "d_Skew",
        "d_Kurt",
        "d_Energy",
        "f_Mean",
        "f_Std",
        "f_P10",
        "f_P90",
        "f_Skew",
        "f_Kurt",
        "f_Energy",
        "Dist2Edge",
        "CentroidShift",
        "Frag",
    ]
    return {f"{prefix}_{k}": 0.0 for k in keys}


def extract_single_case_features(case_id: str, d_path: str, f_path: str, roi_path: str, label_map: List[Tuple[int, int]]) -> Optional[Dict]:
    try:
        roi, img_d, img_f, voxel_sizes, voxel_vol_mm3 = load_case_arrays(d_path, f_path, roi_path)
    except Exception:
        return None

    valid_labels = [x[0] for x in label_map]
    whole_mask = np.isin(roi, valid_labels)
    whole_voxels = int(np.sum(whole_mask))
    if whole_voxels == 0:
        return None

    whole_center = np.array(center_of_mass(whole_mask))
    dist_map = distance_transform_edt(whole_mask, sampling=voxel_sizes)

    features = {}
    for label_id, cluster_id in label_map:
        prefix = f"h{cluster_id}"
        mask = roi == label_id
        n_vox = int(np.sum(mask))
        if n_vox == 0:
            features.update(empty_feature_row(prefix))
            continue

        volume_mm3 = n_vox * voxel_vol_mm3
        features[f"{prefix}_Vol"] = volume_mm3 / 1000.0
        features[f"{prefix}_Ratio"] = n_vox / whole_voxels

        if n_vox > 10:
            try:
                verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=voxel_sizes)
                area = mesh_surface_area(verts, faces)
                sphericity = (np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)) / area
            except Exception:
                sphericity = 0.0
        else:
            sphericity = 0.0
        features[f"{prefix}_Sphericity"] = float(sphericity)

        _, num_components = label(mask)
        features[f"{prefix}_Frag"] = float(num_components)
        features[f"{prefix}_Dist2Edge"] = float(np.mean(dist_map[mask]))

        center_c = np.array(center_of_mass(mask))
        shift_mm = np.sqrt(np.sum(((center_c - whole_center) * voxel_sizes) ** 2))
        features[f"{prefix}_CentroidShift"] = float(shift_mm)

        vals_d = img_d[mask]
        vals_d = vals_d[np.isfinite(vals_d)]
        if vals_d.size > 0:
            features[f"{prefix}_d_Mean"] = float(np.mean(vals_d))
            features[f"{prefix}_d_Std"] = float(np.std(vals_d))
            features[f"{prefix}_d_P10"] = float(np.percentile(vals_d, 10))
            features[f"{prefix}_d_P90"] = float(np.percentile(vals_d, 90))
            features[f"{prefix}_d_Skew"] = float(stats.skew(vals_d))
            features[f"{prefix}_d_Kurt"] = float(stats.kurtosis(vals_d))
            features[f"{prefix}_d_Energy"] = float(np.sum(vals_d ** 2))
        else:
            features.update({f"{prefix}_d_{k}": 0.0 for k in ["Mean", "Std", "P10", "P90", "Skew", "Kurt", "Energy"]})

        vals_f = img_f[mask]
        vals_f = vals_f[np.isfinite(vals_f)]
        if vals_f.size > 0:
            features[f"{prefix}_f_Mean"] = float(np.mean(vals_f))
            features[f"{prefix}_f_Std"] = float(np.std(vals_f))
            features[f"{prefix}_f_P10"] = float(np.percentile(vals_f, 10))
            features[f"{prefix}_f_P90"] = float(np.percentile(vals_f, 90))
            features[f"{prefix}_f_Skew"] = float(stats.skew(vals_f))
            features[f"{prefix}_f_Kurt"] = float(stats.kurtosis(vals_f))
            features[f"{prefix}_f_Energy"] = float(np.sum(vals_f ** 2))
        else:
            features.update({f"{prefix}_f_{k}": 0.0 for k in ["Mean", "Std", "P10", "P90", "Skew", "Kurt", "Energy"]})

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract habitat-level radiomic features from clustered ROI maps.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    num_habitats = int(cfg["features"]["num_habitats"])
    label_map = build_label_map(num_habitats)

    df = pd.read_excel(paths["input_excel"])
    case_id_col = detect_case_id_col(df, cfg["columns"]["case_id_candidates"])
    df = normalize_case_id(df, case_id_col, out_col="CaseID")

    ivim_resampled_dir = paths["ivim_resampled_dir"]
    cluster_output_dir = paths["cluster_output_dir"]
    out_excel = paths["feature_output_excel"]

    rows = []
    missing = 0
    failed = 0
    for case_id in tqdm(df["CaseID"].tolist(), desc="Extracting features"):
        cid = canonical_case_id(case_id)
        d_path = os.path.join(ivim_resampled_dir, f"{cid}_ivim_d.nii.gz")
        f_path = os.path.join(ivim_resampled_dir, f"{cid}_ivim_f.nii.gz")
        roi_path = os.path.join(cluster_output_dir, f"{cid}_cluster_ROI.nii.gz")

        row = {"CaseID": cid}
        if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(roi_path)):
            missing += 1
            rows.append(row)
            continue

        feats = extract_single_case_features(cid, d_path, f_path, roi_path, label_map)
        if feats is None:
            failed += 1
            rows.append(row)
            continue

        row.update(feats)
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    merged = df.copy()
    merged["CaseID"] = merged["CaseID"].map(canonical_case_id)
    feat_df["CaseID"] = feat_df["CaseID"].map(canonical_case_id)
    final_df = pd.merge(merged, feat_df, on="CaseID", how="left")
    final_df.to_excel(out_excel, index=False)

    print(f"Saved feature table: {out_excel}")
    print(f"Cases missing files: {missing}")
    print(f"Cases failed extraction: {failed}")
    print("Pipeline step completed: extract_habitat_features")


if __name__ == "__main__":
    main()
