import os
import json
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage as ndimage
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================= 配置 =================
# 统一使用 F 盘版本路径（可按需改成 H 盘）
EXCEL_PATH = r"F:\RP_multib\副本ALL.xlsx"
RAW_IVIM_DIR = r"F:\RP_multib\ivim"

ROI_INPUT_DIR = r"F:\RP_multib\prediction"
ROI_RESAMPLED_DIR = r"F:\RP_multib\prediction_resampled"
IVIM_RESAMPLED_DIR = r"F:\RP_multib\ivim_resampled"
IVIM_NORMALIZED_DIR = r"F:\RP_multib\ivim_normalized_final3"

COL_CASE_ID = "影像号"
COL_SPLIT = "Dataset_Type"
TARGET_SPACING = [0.5, 0.5, 3.0]

# 物理阈值
D_LIMITS = (0, 3.0e-3)
F_LIMITS = (0, 1.0)

# 单位换算（原始值 -> SI）
D_UNIT_FACTOR = 1000000.0
F_UNIT_FACTOR = 1000.0


def compute_new_size(img, new_spacing):
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    return [int(round(old_size[i] * (old_spacing[i] / new_spacing[i]))) for i in range(3)]


def resample_to_reference(moving, reference, is_label=False):
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    identity = sitk.Transform()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(identity)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)


def clean_data(data, limits, apply_filter=True):
    clipped = np.clip(data, limits[0], limits[1])
    if not apply_filter:
        return clipped

    filtered = np.zeros_like(clipped)
    for z in range(clipped.shape[2]):
        filtered[:, :, z] = ndimage.median_filter(clipped[:, :, z], size=3)
    return filtered


def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def save_nifti(data, affine, header, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(out_img, out_path)


def run_resampling(df):
    print("\n" + "=" * 60)
    print("Step 1/3: 重采样 ROI + IVIM 到 0.5 x 0.5 x 3.0")
    print("=" * 60)

    os.makedirs(ROI_RESAMPLED_DIR, exist_ok=True)
    os.makedirs(IVIM_RESAMPLED_DIR, exist_ok=True)

    allowed_ids = set(df[COL_CASE_ID].astype(str).str.strip().tolist())
    print(f"允许处理的影像号数量: {len(allowed_ids)}")

    def find_roi_path(case_id):
        # 优先匹配无前缀命名
        p1 = os.path.join(ROI_INPUT_DIR, f"{case_id}.nii.gz")
        if os.path.exists(p1):
            return p1
        # 兼容历史命名: xxx_{case_id}.nii.gz
        suffix = f"_{case_id}.nii.gz"
        for fn in os.listdir(ROI_INPUT_DIR):
            if fn.endswith(suffix):
                return os.path.join(ROI_INPUT_DIR, fn)
        return None

    if not os.path.isdir(RAW_IVIM_DIR):
        raise FileNotFoundError(f"原始IVIM目录不存在: {RAW_IVIM_DIR}")

    done = 0
    for case_id in tqdm(os.listdir(RAW_IVIM_DIR), desc="Resample cases"):
        case_path = os.path.join(RAW_IVIM_DIR, case_id)
        if not os.path.isdir(case_path):
            continue
        if str(case_id) not in allowed_ids:
            continue

        roi_path = find_roi_path(case_id)
        t2_path = os.path.join(case_path, "t2w.nii.gz")
        d_path = os.path.join(case_path, "ivim_d.nii.gz")
        f_path = os.path.join(case_path, "ivim_f.nii.gz")

        if not (roi_path and os.path.exists(t2_path) and os.path.exists(d_path) and os.path.exists(f_path)):
            continue

        t2 = sitk.ReadImage(t2_path)
        new_size = compute_new_size(t2, TARGET_SPACING)
        reference = sitk.Resample(
            t2,
            new_size,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            t2.GetOrigin(),
            TARGET_SPACING,
            t2.GetDirection(),
            0,
            t2.GetPixelID()
        )

        roi_img = sitk.ReadImage(roi_path)
        d_img = sitk.ReadImage(d_path)
        f_img = sitk.ReadImage(f_path)

        roi_res = resample_to_reference(roi_img, reference, is_label=True)
        d_res = resample_to_reference(d_img, reference, is_label=False)
        f_res = resample_to_reference(f_img, reference, is_label=False)

        sitk.WriteImage(roi_res, os.path.join(ROI_RESAMPLED_DIR, f"{case_id}.nii.gz"))
        sitk.WriteImage(d_res, os.path.join(IVIM_RESAMPLED_DIR, f"{case_id}_ivim_d.nii.gz"))
        sitk.WriteImage(f_res, os.path.join(IVIM_RESAMPLED_DIR, f"{case_id}_ivim_f.nii.gz"))
        done += 1

    print(f"重采样完成，共处理病例: {done}")


def calculate_training_stats(df):
    print("\n" + "=" * 60)
    print("Step 2/3: 仅用 Training 集 ROI 内体素计算标准化参数")
    print("=" * 60)

    train_df = df[df[COL_SPLIT] == "Training"].copy()
    train_ids = train_df[COL_CASE_ID].astype(str).str.strip().tolist()

    d_pool = []
    f_pool = []

    for case_id in tqdm(train_ids, desc="Collect training voxels"):
        d_path = os.path.join(IVIM_RESAMPLED_DIR, f"{case_id}_ivim_d.nii.gz")
        f_path = os.path.join(IVIM_RESAMPLED_DIR, f"{case_id}_ivim_f.nii.gz")
        roi_path = os.path.join(ROI_RESAMPLED_DIR, f"{case_id}.nii.gz")

        if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(roi_path)):
            continue

        d_data, _, _ = load_nifti(d_path)
        f_data, _, _ = load_nifti(f_path)
        roi_data, _, _ = load_nifti(roi_path)

        mask = roi_data > 0.5
        if np.sum(mask) == 0:
            continue

        d_si = d_data / D_UNIT_FACTOR
        f_si = f_data / F_UNIT_FACTOR
        d_clean = clean_data(d_si, D_LIMITS)
        f_clean = clean_data(f_si, F_LIMITS)

        d_pool.extend(d_clean[mask])
        f_pool.extend(f_clean[mask])

    if len(d_pool) == 0 or len(f_pool) == 0:
        raise RuntimeError("Training 集未收集到有效体素，无法计算统计量。")

    stats = {
        "d_mean": float(np.mean(d_pool)),
        "d_std": float(np.std(d_pool)),
        "f_mean": float(np.mean(f_pool)),
        "f_std": float(np.std(f_pool)),
    }

    print("Training stats (SI units):")
    print(f"  D Mean: {stats['d_mean']:.6f}, Std: {stats['d_std']:.6f}")
    print(f"  f Mean: {stats['f_mean']:.6f}, Std: {stats['f_std']:.6f}")

    os.makedirs(IVIM_NORMALIZED_DIR, exist_ok=True)
    stats_path = os.path.join(IVIM_NORMALIZED_DIR, "stats_reference.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"已保存: {stats_path}")
    return stats


def apply_normalization_all(df, stats):
    print("\n" + "=" * 60)
    print("Step 3/3: 应用到全队列并输出标准化 IVIM 图")
    print("=" * 60)

    case_ids = df[COL_CASE_ID].astype(str).str.strip().tolist()
    os.makedirs(IVIM_NORMALIZED_DIR, exist_ok=True)
    done = 0

    for case_id in tqdm(case_ids, desc="Normalize all cases"):
        d_path = os.path.join(IVIM_RESAMPLED_DIR, f"{case_id}_ivim_d.nii.gz")
        f_path = os.path.join(IVIM_RESAMPLED_DIR, f"{case_id}_ivim_f.nii.gz")
        if not (os.path.exists(d_path) and os.path.exists(f_path)):
            continue

        d_data, aff, hdr = load_nifti(d_path)
        f_data, _, _ = load_nifti(f_path)

        d_si = d_data / D_UNIT_FACTOR
        f_si = f_data / F_UNIT_FACTOR
        d_clean = clean_data(d_si, D_LIMITS)
        f_clean = clean_data(f_si, F_LIMITS)

        d_norm = (d_clean - stats["d_mean"]) / (stats["d_std"] + 1e-8)
        f_norm = (f_clean - stats["f_mean"]) / (stats["f_std"] + 1e-8)

        save_nifti(d_norm, aff, hdr, os.path.join(IVIM_NORMALIZED_DIR, f"{case_id}_ivim_d_norm.nii.gz"))
        save_nifti(f_norm, aff, hdr, os.path.join(IVIM_NORMALIZED_DIR, f"{case_id}_ivim_f_norm.nii.gz"))
        done += 1

    print(f"标准化完成，共输出病例: {done}")


def main():
    print("Loading Excel...")
    df = pd.read_excel(EXCEL_PATH)
    df[COL_CASE_ID] = df[COL_CASE_ID].astype(str).str.strip()

    run_resampling(df)
    stats = calculate_training_stats(df)
    apply_normalization_all(df, stats)
    print("\n✅ 全流程完成：重采样 + 标准化")


if __name__ == "__main__":
    main()
