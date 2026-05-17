import pandas as pd
import numpy as np
import nibabel as nib
import os
from scipy import stats
from scipy.ndimage import distance_transform_edt, center_of_mass, label
from skimage.measure import marching_cubes, mesh_surface_area
from tqdm import tqdm  # 进度条
import warnings

warnings.filterwarnings('ignore') # 忽略一些计算中的除零警告

# ================= 配置区域 =================
# 输入 Excel
EXCEL_PATH = r"F:\RP_multib\副本ALL911.xlsx"

# 影像文件夹
IMG_DIR = r"F:\RP_multib\ivim_resampled"
# ROI 文件夹
ROI_DIR = r"F:\RP_multib\ivim_clustering_results911"

# 输出路径
OUTPUT_PATH = r"F:\RP_multib\ALL911_Radiomics_Features_Corrected1.xlsx"

# ROI 标签定义 (Label 1->Cluster0, Label 2->Cluster1...)
# 格式: (ROI文件中的标签值, 输出特征时的生境编号)
ROI_LABEL_MAP = [(1, 0), (2, 1), (3, 2), (4, 3)]

# ================= 核心计算函数 =================

def extract_features_single_case(case_id, d_path, f_path, roi_path):
    """提取单个病例的所有特征"""
    
    # 1. 加载数据
    try:
        nii_roi = nib.load(roi_path)
        img_roi = nii_roi.get_fdata().astype(int)
        
        # 获取体素尺寸 (用于计算真实的体积和距离)
        voxel_sizes = nii_roi.header.get_zooms()[:3] # (x, y, z)
        voxel_vol = np.prod(voxel_sizes) # 单个体素的体积 mm^3
        
        img_d = nib.load(d_path).get_fdata()
        img_f = nib.load(f_path).get_fdata()
    except Exception as e:
        print(f"❌ [Error] {case_id}: 文件加载失败 - {e}")
        return None

    # 2. 定义全肿瘤区域 (排除背景0)
    # [修改点1] 全肿瘤是标签 1, 2, 3, 4 的并集
    valid_labels = [label for label, _ in ROI_LABEL_MAP]
    mask_whole_tumor = np.isin(img_roi, valid_labels)
    
    # 如果全肿瘤体积为0，跳过
    if np.sum(mask_whole_tumor) == 0:
        return None

    # 3. 预计算全肿瘤的空间属性 (用于 Morph/Spatial 特征)
    vol_pixels_whole = np.sum(mask_whole_tumor)
    
    # A. 全肿瘤几何中心
    center_whole = np.array(center_of_mass(mask_whole_tumor))
    
    # B. 全肿瘤距离图
    # [解释] distance_transform_edt 计算点到背景(0)的距离
    # 在 mask_whole_tumor 中，背景是False(0)，肿瘤是True(1)
    # 结果图中，肿瘤边缘的值小，中心的值大
    dist_map_whole = distance_transform_edt(mask_whole_tumor, sampling=voxel_sizes)

    # 存储特征的字典
    features = {}

    # 4. 遍历每个生境提取特征
    for label_id, cluster_id in ROI_LABEL_MAP:
        prefix = f"h{cluster_id}" # [修改点1] 输出 h0, h1... 对应 Label 1, 2...
        
        # [修改点1] 提取对应的标签值 (1,2,3,4)
        mask_c = (img_roi == label_id)
        vol_pixels_c = np.sum(mask_c)
        
        # --- 情况 A: 该生境不存在 (Volume=0) ---
        if vol_pixels_c == 0:
            # 填 0 (注意增加了 P90 和 P10)
            zero_cols = [
                'Vol', 'Ratio', 'Sphericity', 
                'd_Mean', 'd_Std', 'd_P10', 'd_P90', 'd_Skew', 'd_Kurt', 'd_Energy',
                'f_Mean', 'f_Std', 'f_P10', 'f_P90', 'f_Skew', 'f_Kurt', 'f_Energy',
                'Dist2Edge', 'CentroidShift', 'Frag'
            ]
            for col in zero_cols: features[f"{prefix}_{col}"] = 0
            continue

        # --- 情况 B: 生境存在 ---
        
        # === 1. 形态学 (Morphology) ===
        vol_mm3 = vol_pixels_c * voxel_vol
        features[f"{prefix}_Vol"] = vol_mm3 / 1000.0 # ml
        features[f"{prefix}_Ratio"] = vol_pixels_c / vol_pixels_whole
        
        # 球形度
        try:
            if vol_pixels_c > 10:
                verts, faces, _, _ = marching_cubes(mask_c, level=0.5, spacing=voxel_sizes)
                surf_area = mesh_surface_area(verts, faces)
                sphericity = (np.pi**(1/3) * (6 * vol_mm3)**(2/3)) / surf_area
            else:
                sphericity = 0 
        except:
            sphericity = 0
        features[f"{prefix}_Sphericity"] = sphericity

        # === 2. 空间/拓扑 (Spatial) ===
        labeled_array, num_features = label(mask_c)
        features[f"{prefix}_Frag"] = num_features
        
        dists_in_habitat = dist_map_whole[mask_c]
        features[f"{prefix}_Dist2Edge"] = np.mean(dists_in_habitat)
        
        center_c = np.array(center_of_mass(mask_c))
        shift_mm = np.sqrt(np.sum(((center_c - center_whole) * voxel_sizes)**2))
        features[f"{prefix}_CentroidShift"] = shift_mm

        # === 3. 一阶统计 (D图) ===
        vals_d = img_d[mask_c]
        vals_d = vals_d[np.isfinite(vals_d)]
        
        if len(vals_d) > 0:
            features[f"{prefix}_d_Mean"] = np.mean(vals_d)
            features[f"{prefix}_d_Std"] = np.std(vals_d)
            features[f"{prefix}_d_P10"] = np.percentile(vals_d, 10) # [修改点2] D P10
            features[f"{prefix}_d_P90"] = np.percentile(vals_d, 90) # [修改点2] D P90
            features[f"{prefix}_d_Skew"] = stats.skew(vals_d)
            features[f"{prefix}_d_Kurt"] = stats.kurtosis(vals_d)
            features[f"{prefix}_d_Energy"] = np.sum(vals_d**2)
        else:
            for k in ['Mean', 'Std', 'P10', 'P90', 'Skew', 'Kurt', 'Energy']: features[f"{prefix}_d_{k}"] = 0

        # === 4. 一阶统计 (f图) ===
        vals_f = img_f[mask_c]
        vals_f = vals_f[np.isfinite(vals_f)]
        
        if len(vals_f) > 0:
            features[f"{prefix}_f_Mean"] = np.mean(vals_f)
            features[f"{prefix}_f_Std"] = np.std(vals_f)
            features[f"{prefix}_f_P10"] = np.percentile(vals_f, 10) # [修改点2] f P10
            features[f"{prefix}_f_P90"] = np.percentile(vals_f, 90) # [修改点2] f P90
            features[f"{prefix}_f_Skew"] = stats.skew(vals_f)
            features[f"{prefix}_f_Kurt"] = stats.kurtosis(vals_f)
            features[f"{prefix}_f_Energy"] = np.sum(vals_f**2)
        else:
            for k in ['Mean', 'Std', 'P10', 'P90', 'Skew', 'Kurt', 'Energy']: features[f"{prefix}_f_{k}"] = 0

    return features

# ================= 主流程 =================

def main():
    print(f"📂 读取 Excel: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    
    results = []
    error_log = []
    
    print("🚀 开始提取特征...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        case_id = str(row['CaseID']).strip()
        
        d_path = os.path.join(IMG_DIR, f"{case_id}_ivim_d.nii.gz")
        f_path = os.path.join(IMG_DIR, f"{case_id}_ivim_f.nii.gz")
        roi_path = os.path.join(ROI_DIR, f"{case_id}_cluster_ROI.nii.gz")
        
        if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(roi_path)):
            error_log.append(f"{case_id}: 文件缺失")
            results.append({'CaseID': case_id}) 
            continue
            
        case_feats = extract_features_single_case(case_id, d_path, f_path, roi_path)
        
        if case_feats:
            row_data = {'CaseID': case_id}
            row_data.update(case_feats)
            results.append(row_data)
        else:
            error_log.append(f"{case_id}: 提取失败或ROI为空")
            results.append({'CaseID': case_id})

    print("-" * 50)
    print("💾 正在保存结果...")
    
    df_feats = pd.DataFrame(results)
    
    df['CaseID'] = df['CaseID'].astype(str).str.strip()
    df_feats['CaseID'] = df_feats['CaseID'].astype(str).str.strip()
    # 合并
    df_final = pd.merge(df, df_feats, on='CaseID', how='left')
    
    try:
        df_final.to_excel(OUTPUT_PATH, index=False)
        print(f"✅ 成功! 文件已保存至: {OUTPUT_PATH}")
    except PermissionError:
        print("❌ 保存失败: 请关闭目标 Excel 文件后重试。")
        
    if error_log:
        print(f"\n⚠️ 共有 {len(error_log)} 例处理异常:")
        # print(error_log[:5])

if __name__ == "__main__":
    main()