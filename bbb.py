import os
import pandas as pd
import numpy as np
import nibabel as nib
import json
import scipy.ndimage as ndimage

# ================= 配置路径 =================
# 注意：这里读取的是刚刚生成的带标签的表格！
EXCEL_PATH = r"F:\RP_multib\副本ALL.xlsx" 

IVIM_DIR = r"F:\RP_multib\ivim_resampled"
ROI_DIR = r"F:\RP_multib\prediction_resampled"
OUTPUT_DIR = r"F:\RP_multib\ivim_normalized_final3" 

# 列名
COL_CASE_ID = "影像号"
COL_SPLIT = "Dataset_Type" # 我们刚生成的那一列

# 物理阈值
D_LIMITS = (0, 3.0e-3) 
f_LIMITS = (0, 1.0)    

# ⭐ 单位换算因子 - 必须使用这些因子将原始数据转换为 SI 单位
# ----------------------------------------------------
D_UNIT_FACTOR = 1000000.0 # 假设原始 D 值是 10^6 乘以 SI 单位
f_UNIT_FACTOR = 1000.0    # 假设原始 f 值是 1000 乘以 SI 单位
# ================= 功能函数 (保持不变) =================

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header

def save_nifti(data, affine, header, out_path):
    new_img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(new_img, out_path)

def clean_data(data, limits, apply_filter=True):
    # 物理截断
    data_clean = np.clip(data, limits[0], limits[1])
    # 中值滤波
    if apply_filter:
        filtered = np.zeros_like(data_clean)
        for z in range(data_clean.shape[2]):
            filtered[:,:,z] = ndimage.median_filter(data_clean[:,:,z], size=3)
        return filtered
    else:
        return data_clean

def get_paths(case_id):
    d_path = os.path.join(IVIM_DIR, f"{case_id}_ivim_d.nii.gz")
    f_path = os.path.join(IVIM_DIR, f"{case_id}_ivim_f.nii.gz")
    roi_path = os.path.join(ROI_DIR, f"{case_id}.nii.gz")
    return d_path, f_path, roi_path

# ================= 核心修改：只用 Training 集计算统计量 =================

def calculate_training_stats(df):
    print(f"\n{'='*40}")
    print(f"Step 1: Calculating Stats from 'Training' Subset Only")
    print(f"{'='*40}")
    
    train_ids = df[df[COL_SPLIT] == 'Training'][COL_CASE_ID].tolist()
    d_pool = []
    f_pool = []
    
    for case_id in train_ids:
        d_p, f_p, r_p = get_paths(case_id)
        if not (os.path.exists(d_p) and os.path.exists(f_p) and os.path.exists(r_p)): continue
            
        d_data, _, _ = load_nifti(d_p)
        f_data, _, _ = load_nifti(f_p)
        roi_data, _, _ = load_nifti(r_p)
        mask = roi_data > 0.5
        if np.sum(mask) == 0: continue
        
        # ----------------------------------------------------
        # ⭐ 关键修改：单位换算 (在清洗前进行!)
        # ----------------------------------------------------
        d_data_si = d_data / D_UNIT_FACTOR
        f_data_si = f_data / f_UNIT_FACTOR
        
        # 2. 清洗 (现在使用正确单位的 D_LIMITS 和 f_LIMITS)
        d_clean = clean_data(d_data_si, D_LIMITS)
        f_clean = clean_data(f_data_si, f_LIMITS)
        
        # 收集
        d_pool.extend(d_clean[mask])
        f_pool.extend(f_clean[mask])
        
    # 计算统计量
    # ... (Stats calculation logic remains the same) ...

    stats = {
        'd_mean': float(np.mean(d_pool)),
        'd_std': float(np.std(d_pool)),
        'f_mean': float(np.mean(f_pool)),
        'f_std': float(np.std(f_pool))
    }
    
    print("\n✅ Training Set Statistics Calculated (SI Units):")
    print(f"  D Mean: {stats['d_mean']:.6f} | Std: {stats['d_std']:.6f}")
    print(f"  f Mean: {stats['f_mean']:.6f} | Std: {stats['f_std']:.6f}")
    
    # 保存参数供后续查看
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, 'stats_reference.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    return stats

# ================= 应用到全集 (保持不变) =================

def apply_normalization_to_all(df, stats):
    # ... (应用标准化函数的开头保持不变) ...
    
    for idx, case_id in enumerate(df[COL_CASE_ID].tolist()):
        d_p, f_p, r_p = get_paths(case_id)
        if not (os.path.exists(d_p) and os.path.exists(f_p)): continue
            
        d_data, aff, hdr = load_nifti(d_p)
        f_data, _, _ = load_nifti(f_p)
        
        # ----------------------------------------------------
        # ⭐ 关键修改：应用时也要先换算单位！
        # ----------------------------------------------------
        d_data_si = d_data / D_UNIT_FACTOR
        f_data_si = f_data / f_UNIT_FACTOR
        
        # 2. 清洗 (使用 SI 单位数据)
        d_clean = clean_data(d_data_si, D_LIMITS)
        f_clean = clean_data(f_data_si, f_LIMITS)
        
        # 3. 标准化 (使用 SI 单位数据和 SI 单位的 stats)
        d_norm = (d_clean - stats['d_mean']) / (stats['d_std'] + 1e-8)
        f_norm = (f_clean - stats['f_mean']) / (stats['f_std'] + 1e-8)
        
        # 4. 保存
        save_nifti(d_norm, aff, hdr, os.path.join(OUTPUT_DIR, f"{case_id}_ivim_d_norm.nii.gz"))
        save_nifti(f_norm, aff, hdr, os.path.join(OUTPUT_DIR, f"{case_id}_ivim_f_norm.nii.gz"))

    print("Done.")

# ================= 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print("Loading Labeled Excel...")
    df = pd.read_excel(EXCEL_PATH)
    df[COL_CASE_ID] = df[COL_CASE_ID].astype(str).str.strip()
    
    # 1. 计算 Training 集的均值方差
    stats = calculate_training_stats(df)
    
    # 2. 应用到所有人
    apply_normalization_to_all(df, stats)

if __name__ == "__main__":
    main()