import argparse
import os
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
from tqdm import tqdm

from pipeline_utils import detect_case_id_col, ensure_dir, load_config, normalize_case_id


def canonical_case_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def compute_new_size(img: sitk.Image, new_spacing: Tuple[float, float, float]):
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    return [int(round(old_size[i] * (old_spacing[i] / new_spacing[i]))) for i in range(3)]


def make_reference(t2_img: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
    new_size = compute_new_size(t2_img, target_spacing)
    return sitk.Resample(
        t2_img,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        t2_img.GetOrigin(),
        target_spacing,
        t2_img.GetDirection(),
        0,
        t2_img.GetPixelID(),
    )


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, is_label: bool = False) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)


def load_nifti(path: str):
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def save_nifti(data: np.ndarray, affine, header, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    out = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(out, out_path)


def clean_volume(data: np.ndarray, limits: Tuple[float, float], median_filter_size: int = 3) -> np.ndarray:
    clipped = np.clip(data, limits[0], limits[1])
    filtered = np.zeros_like(clipped)
    for z in range(clipped.shape[2]):
        filtered[:, :, z] = ndimage.median_filter(clipped[:, :, z], size=median_filter_size)
    return filtered


def find_roi_path(roi_input_dir: str, case_id: str) -> Optional[str]:
    preferred = os.path.join(roi_input_dir, f"{case_id}.nii.gz")
    if os.path.exists(preferred):
        return preferred

    suffix = f"_{case_id}.nii.gz"
    for fn in os.listdir(roi_input_dir):
        if fn.endswith(suffix):
            return os.path.join(roi_input_dir, fn)
    return None


def run_resampling(df: pd.DataFrame, cfg: dict) -> None:
    paths = cfg["paths"]
    preprocess_cfg = cfg["preprocess"]
    target_spacing = tuple(preprocess_cfg["target_spacing"])

    raw_ivim_dir = paths["raw_ivim_dir"]
    roi_input_dir = paths["roi_input_dir"]
    roi_resampled_dir = paths["roi_resampled_dir"]
    ivim_resampled_dir = paths["ivim_resampled_dir"]

    ensure_dir(roi_resampled_dir)
    ensure_dir(ivim_resampled_dir)

    allowed_ids = set(df["CaseID"].map(canonical_case_id).tolist())
    if not os.path.isdir(raw_ivim_dir):
        raise FileNotFoundError(f"Raw IVIM directory not found: {raw_ivim_dir}")

    processed = 0
    for case_folder in tqdm(os.listdir(raw_ivim_dir), desc="Resampling cases"):
        case_id = canonical_case_id(case_folder)
        if case_id not in allowed_ids:
            continue

        case_path = os.path.join(raw_ivim_dir, case_folder)
        if not os.path.isdir(case_path):
            continue

        t2_path = os.path.join(case_path, "t2w.nii.gz")
        d_path = os.path.join(case_path, "ivim_d.nii.gz")
        f_path = os.path.join(case_path, "ivim_f.nii.gz")
        roi_path = find_roi_path(roi_input_dir, case_id)

        if not (os.path.exists(t2_path) and os.path.exists(d_path) and os.path.exists(f_path) and roi_path):
            continue

        t2_img = sitk.ReadImage(t2_path)
        ref = make_reference(t2_img, target_spacing)

        roi_img = sitk.ReadImage(roi_path)
        d_img = sitk.ReadImage(d_path)
        f_img = sitk.ReadImage(f_path)

        roi_res = resample_to_reference(roi_img, ref, is_label=True)
        d_res = resample_to_reference(d_img, ref, is_label=False)
        f_res = resample_to_reference(f_img, ref, is_label=False)

        sitk.WriteImage(roi_res, os.path.join(roi_resampled_dir, f"{case_id}.nii.gz"))
        sitk.WriteImage(d_res, os.path.join(ivim_resampled_dir, f"{case_id}_ivim_d.nii.gz"))
        sitk.WriteImage(f_res, os.path.join(ivim_resampled_dir, f"{case_id}_ivim_f.nii.gz"))
        processed += 1

    print(f"Resampling completed. Cases processed: {processed}")


def calculate_training_stats(df: pd.DataFrame, cfg: dict) -> dict:
    paths = cfg["paths"]
    preprocess_cfg = cfg["preprocess"]
    split_col = cfg["columns"]["split_col"]
    training_value = cfg["dataset_values"]["training"]

    d_limits = tuple(preprocess_cfg["d_limits"])
    f_limits = tuple(preprocess_cfg["f_limits"])
    d_unit_factor = float(preprocess_cfg["d_unit_factor"])
    f_unit_factor = float(preprocess_cfg["f_unit_factor"])
    median_filter_size = int(preprocess_cfg["median_filter_size"])

    ivim_resampled_dir = paths["ivim_resampled_dir"]
    roi_resampled_dir = paths["roi_resampled_dir"]
    ivim_normalized_dir = paths["ivim_normalized_dir"]
    ensure_dir(ivim_normalized_dir)

    train_df = df[df[split_col].astype(str).str.strip() == training_value].copy()
    d_pool = []
    f_pool = []

    for case_id in tqdm(train_df["CaseID"].tolist(), desc="Collecting training voxels"):
        cid = canonical_case_id(case_id)
        d_path = os.path.join(ivim_resampled_dir, f"{cid}_ivim_d.nii.gz")
        f_path = os.path.join(ivim_resampled_dir, f"{cid}_ivim_f.nii.gz")
        roi_path = os.path.join(roi_resampled_dir, f"{cid}.nii.gz")
        if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(roi_path)):
            continue

        d_data, _, _ = load_nifti(d_path)
        f_data, _, _ = load_nifti(f_path)
        roi_data, _, _ = load_nifti(roi_path)

        mask = roi_data > 0.5
        if np.sum(mask) == 0:
            continue

        d_si = d_data / d_unit_factor
        f_si = f_data / f_unit_factor
        d_clean = clean_volume(d_si, d_limits, median_filter_size)
        f_clean = clean_volume(f_si, f_limits, median_filter_size)

        d_pool.extend(d_clean[mask])
        f_pool.extend(f_clean[mask])

    if len(d_pool) == 0 or len(f_pool) == 0:
        raise RuntimeError("No valid training voxels found for normalization statistics.")

    stats = {
        "d_mean": float(np.mean(d_pool)),
        "d_std": float(np.std(d_pool)),
        "f_mean": float(np.mean(f_pool)),
        "f_std": float(np.std(f_pool)),
    }

    stats_path = os.path.join(ivim_normalized_dir, "stats_reference.json")
    import json

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Training normalization statistics saved:", stats_path)
    print(stats)
    return stats


def apply_normalization_to_all(df: pd.DataFrame, stats: dict, cfg: dict) -> None:
    paths = cfg["paths"]
    preprocess_cfg = cfg["preprocess"]

    d_limits = tuple(preprocess_cfg["d_limits"])
    f_limits = tuple(preprocess_cfg["f_limits"])
    d_unit_factor = float(preprocess_cfg["d_unit_factor"])
    f_unit_factor = float(preprocess_cfg["f_unit_factor"])
    median_filter_size = int(preprocess_cfg["median_filter_size"])

    ivim_resampled_dir = paths["ivim_resampled_dir"]
    ivim_normalized_dir = paths["ivim_normalized_dir"]
    ensure_dir(ivim_normalized_dir)

    done = 0
    for case_id in tqdm(df["CaseID"].tolist(), desc="Normalizing all cases"):
        cid = canonical_case_id(case_id)
        d_path = os.path.join(ivim_resampled_dir, f"{cid}_ivim_d.nii.gz")
        f_path = os.path.join(ivim_resampled_dir, f"{cid}_ivim_f.nii.gz")
        if not (os.path.exists(d_path) and os.path.exists(f_path)):
            continue

        d_data, aff, hdr = load_nifti(d_path)
        f_data, _, _ = load_nifti(f_path)

        d_si = d_data / d_unit_factor
        f_si = f_data / f_unit_factor
        d_clean = clean_volume(d_si, d_limits, median_filter_size)
        f_clean = clean_volume(f_si, f_limits, median_filter_size)

        d_norm = (d_clean - stats["d_mean"]) / (stats["d_std"] + 1e-8)
        f_norm = (f_clean - stats["f_mean"]) / (stats["f_std"] + 1e-8)

        save_nifti(d_norm, aff, hdr, os.path.join(ivim_normalized_dir, f"{cid}_ivim_d_norm.nii.gz"))
        save_nifti(f_norm, aff, hdr, os.path.join(ivim_normalized_dir, f"{cid}_ivim_f_norm.nii.gz"))
        done += 1

    print(f"Normalization completed. Cases exported: {done}")


def main():
    parser = argparse.ArgumentParser(description="Resample and normalize IVIM volumes for habitat analysis.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_excel = cfg["paths"]["input_excel"]

    df = pd.read_excel(input_excel)
    case_id_col = detect_case_id_col(df, cfg["columns"]["case_id_candidates"])
    df = normalize_case_id(df, case_id_col, out_col="CaseID")

    run_resampling(df, cfg)
    stats = calculate_training_stats(df, cfg)
    apply_normalization_to_all(df, stats, cfg)
    print("Pipeline step completed: preprocess_resample_normalize")


if __name__ == "__main__":
    main()
