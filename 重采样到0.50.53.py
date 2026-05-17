import os
import SimpleITK as sitk
import pandas as pd

# ========= 配置 ==========
target_spacing = [0.5, 0.5, 3.0]

ivim_root = r"H:\RP_multib"
ivim_folders = ["pros_siemens", "retro_siemens", "retro_GE", "retro_zoomit", "RP_new"]

roi_input_dir = r"H:\RP_multib\prediction"
roi_output_dir = r"H:\RP_multib\prediction_resampled"
ivim_output_dir = r"H:\RP_multib\ivim_resampled"

os.makedirs(roi_output_dir, exist_ok=True)
os.makedirs(ivim_output_dir, exist_ok=True)

# ========= 读取 Excel 中允许处理的影像号 ==========
excel_path = r"H:\RP_multib\ALL.xlsx"
df = pd.read_excel(excel_path, sheet_name=0)

allowed_ids = set(df["影像号"].astype(str).tolist())   # 转为字符串，避免 123 vs "123" 不匹配

print(f"允许处理的影像号数量：{len(allowed_ids)}")


def compute_new_size(img, new_spacing):
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_size = [
        int(round(old_size[i] * (old_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    return new_size


def resample(moving, reference, is_label=False):
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    identity = sitk.Transform()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(identity)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(moving)


# ============= 主流程 =============
for folder in ivim_folders:
    folder_path = os.path.join(ivim_root, folder)

    for case_id in os.listdir(folder_path):
        case_path = os.path.join(folder_path, case_id)
        if not os.path.isdir(case_path):
            continue

        # （新增）如果影像号不在 Excel 列表中 → 直接跳过
        if str(case_id) not in allowed_ids:
            print(f"不在Excel列表中，跳过：{case_id}")
            continue

        # ROI 文件名如：pros_siemens_123456.nii.gz
        roi_name = f"{folder}_{case_id}.nii.gz"
        roi_path = os.path.join(roi_input_dir, roi_name)

        if not os.path.exists(roi_path):
            print("ROI 不存在，跳过：", roi_name)
            continue

        # T2 图作为参考
        t2_path = os.path.join(case_path, "t2w.nii.gz")
        if not os.path.exists(t2_path):
            print("T2 不存在，跳过：", t2_path)
            continue

        # 创建 reference image
        t2 = sitk.ReadImage(t2_path)
        new_size = compute_new_size(t2, target_spacing)

        reference = sitk.Resample(
            t2,
            new_size,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            t2.GetOrigin(),
            target_spacing,
            t2.GetDirection(),
            0,
            t2.GetPixelID()
        )

        # ---------- 重采样 ROI ----------
        roi = sitk.ReadImage(roi_path)
        roi_res = resample(roi, reference, is_label=True)

        roi_out = os.path.join(roi_output_dir, f"{case_id}.nii.gz")
        sitk.WriteImage(roi_res, roi_out)

        # ---------- 重采样 IVIM ----------
        d_path = os.path.join(case_path, "ivim_d.nii.gz")
        f_path = os.path.join(case_path, "ivim_f.nii.gz")

        if not (os.path.exists(d_path) and os.path.exists(f_path)):
            print("IVIM 缺失，跳过：", case_id)
            continue

        d_img = sitk.ReadImage(d_path)
        f_img = sitk.ReadImage(f_path)

        d_res = resample(d_img, reference)
        f_res = resample(f_img, reference)

        ivim_d_out = os.path.join(ivim_output_dir, f"{case_id}_ivim_d.nii.gz")
        ivim_f_out = os.path.join(ivim_output_dir, f"{case_id}_ivim_f.nii.gz")

        sitk.WriteImage(d_res, ivim_d_out)
        sitk.WriteImage(f_res, ivim_f_out)

        print(f"完成 case：{case_id}")

print("全部完成！")
