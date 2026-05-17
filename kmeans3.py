import os
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ================= 核心配置 =================
EXCEL_PATH = r"F:\RP_multib\副本ALL.xlsx"
NORM_IVIM_DIR = r"F:\RP_multib\ivim_normalized_final3"
ROI_DIR = r"F:\RP_multib\prediction_resampled"
OUTPUT_DIR = r"F:\RP_multib\ivim_clustering_results3"
STATS_PATH = os.path.join(NORM_IVIM_DIR, 'stats_reference.json')

N_TARGET = 5000
RANDOM_SEED = 42

# 影像号和分组列名
COL_CASE_ID = "影像号"
COL_SPLIT = "Dataset_Type"

# 设定并行工作的核心数 (建议设为 8 以平衡负载)
MAX_WORKERS = 12 

# ================= Worker 函数 (必须放在顶级) =================

def worker_load_and_sample(row_data):
    """阶段1 Worker: 负责加载数据并进行降采样"""
    case_id, dataset_type, n_target, seed, d_path, f_path, r_path = row_data
    
    if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(r_path)):
        return None

    try:
        d_norm = nib.load(d_path).get_fdata()
        f_norm = nib.load(f_path).get_fdata()
        roi_data = nib.load(r_path).get_fdata()
        
        mask = roi_data > 0.5
        indices = np.where(mask)
        N_actual = len(indices[0])
        
        if N_actual < 10: return None

        d_pixels = d_norm[mask].flatten()
        f_pixels = f_norm[mask].flatten()
        
        # 降采样逻辑
        if N_actual <= n_target:
            sampled_indices = np.arange(N_actual)
        else:
            all_indices = np.arange(N_actual)
            rng = np.random.default_rng(seed + hash(case_id) % 10000)
            rng.shuffle(all_indices) 
            R = N_actual / n_target
            sampled_indices = all_indices[np.round(np.arange(0, N_actual, R)).astype(int)] 
            sampled_indices = sampled_indices[:n_target]

        d_sampled = d_pixels[sampled_indices]
        f_sampled = f_pixels[sampled_indices]
        
        return {
            'D_norm': d_sampled, 
            'f_norm': f_sampled
        }
    except Exception:
        return None

def worker_predict_and_save(args):
    """阶段3 Worker: 负责应用模型、生成ROI、保存文件"""
    row, kmeans_centers, mu_d, std_d, mu_f, std_f, n_clusters, output_dir, norm_dir, roi_dir = args
    
    case_id = row[COL_CASE_ID]
    dataset_type = row[COL_SPLIT]
    
    d_path = os.path.join(norm_dir, f"{case_id}_ivim_d_norm.nii.gz")
    f_path = os.path.join(norm_dir, f"{case_id}_ivim_f_norm.nii.gz")
    r_path = os.path.join(roi_dir, f"{case_id}.nii.gz")

    if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(r_path)):
        return None

    try:
        d_norm_img = nib.load(d_path)
        d_norm_data = d_norm_img.get_fdata()
        affine = d_norm_img.affine
        f_norm_data = nib.load(f_path).get_fdata()
        roi_data = nib.load(r_path).get_fdata()
        
        mask = roi_data > 0.5
        if np.sum(mask) < 10: return None
        
        d_pixels = d_norm_data[mask]
        f_pixels = f_norm_data[mask]
        X_case = np.column_stack((d_pixels, f_pixels))
        
        # 手动计算距离进行预测 (避免传递 KMeans 对象)
        distances = np.linalg.norm(X_case[:, np.newaxis, :] - kmeans_centers, axis=2)
        labels = np.argmin(distances, axis=1)
        
        clustered_roi_map = np.zeros_like(d_norm_data, dtype=np.uint8)
        clustered_roi_map[mask] = labels + 1 
        
        out_path = os.path.join(output_dir, f"{case_id}_cluster_ROI.nii.gz")
        new_img = nib.Nifti1Image(clustered_roi_map, affine)
        nib.save(new_img, out_path)
        
        voxel_volume = np.prod(affine[0:3, 0:3].diagonal()) / 1000.0
        total_voxels = len(labels)
        
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        result = {'CaseID': case_id, 'Dataset_Type': dataset_type, 'Total_Volume_ml': total_voxels * voxel_volume}
        
        for i in range(n_clusters):
            count = cluster_counts.get(i, 0)
            result[f'Cluster_{i}_Ratio'] = count / total_voxels
            result[f'Cluster_{i}_Volume_ml'] = count * voxel_volume
            
        return result
    except Exception:
        return None

# ================= 管理函数 =================

def prepare_data_multiprocess(df, dataset_filter='Training'):
    """多进程数据加载"""
    if dataset_filter == 'Training':
        df_target = df[df[COL_SPLIT] == 'Training']
    elif dataset_filter == 'All':
        df_target = df 
    else:
        df_target = df[df[COL_SPLIT] == dataset_filter]

    print(f"\n🚀 Multiprocessing: Loading & Sampling ({len(df_target)} cases)...")
    
    tasks = []
    for _, row in df_target.iterrows():
        case_id = row[COL_CASE_ID]
        d_p = os.path.join(NORM_IVIM_DIR, f"{case_id}_ivim_d_norm.nii.gz")
        f_p = os.path.join(NORM_IVIM_DIR, f"{case_id}_ivim_f_norm.nii.gz")
        r_p = os.path.join(ROI_DIR, f"{case_id}.nii.gz")
        tasks.append((case_id, row[COL_SPLIT], N_TARGET, RANDOM_SEED, d_p, f_p, r_p))

    all_pixel_info = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker_load_and_sample, t): t for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="  CPU Loading"):
            res = future.result()
            if res is not None:
                df_chunk = pd.DataFrame({'D_norm': res['D_norm'], 'f_norm': res['f_norm']})
                all_pixel_info.append(df_chunk)

    if not all_pixel_info:
        return None
    
    print("  Stacking dataframes (this may take a moment)...")
    pixel_df = pd.concat(all_pixel_info, ignore_index=True)
    feature_matrix = pixel_df[['D_norm', 'f_norm']].values
    print(f"  Total {len(feature_matrix)} pixels ready.")
    
    return feature_matrix

def find_optimal_k_automatic(k_values, inertias):
    """
    使用 Kneedle 算法 (最大距离法) 自动寻找肘部点
    """
    n_points = len(k_values)
    all_coord = np.vstack((k_values, inertias)).T
    
    # 1. 坐标归一化 (因为 K值很小，Inertia值很大，必须归一化才能计算距离)
    # min-max normalization
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # 归一化处理
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    
    # 2. 构建起点到终点的直线向量
    # 起点 (0, 1) -> 因为 inertia 是下降的，归一化后起点 inertia 最大
    # 终点 (1, 0)
    vec_line = np.array([1, -1]) # 终点(1,0) - 起点(0,1)
    
    # 3. 计算每个点到直线的距离
    # 向量计算：点到直线的距离
    distances = []
    for i in range(n_points):
        point_vec = np.array([k_norm[i], inertia_norm[i]]) - np.array([0, 1])
        # 叉乘计算距离 (二维叉乘)
        dist = np.abs(np.cross(vec_line, point_vec)) / np.linalg.norm(vec_line)
        distances.append(dist)
        
    # 4. 找到最大距离对应的索引
    best_idx = np.argmax(distances)
    return k_values[best_idx]

def determine_optimal_k(feature_matrix, k_max=8):
    """
    计算 WCSS 并自动返回最佳 K 值，无需人工干预
    """
    print(f"\n{'='*50}")
    print(f"⚙️ Determining Optimal K Automatically (K=2 to {k_max})")
    print(f"{'='*50}")
    
    inertias = []
    k_range = list(range(2, k_max + 1))
    
    # 计算 WCSS
    for k in tqdm(k_range, desc="  Fitting KMeans"):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(feature_matrix)
        inertias.append(kmeans.inertia_)
        
    # 转换为 numpy 数组以便计算
    k_range_arr = np.array(k_range)
    inertias_arr = np.array(inertias)
    
    # --- 自动计算最佳 K ---
    optimal_k = find_optimal_k_automatic(k_range_arr, inertias_arr)
    
    # 绘制曲线并标记自动选择的点
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o', linestyle='-', color='b', label='Inertia (WCSS)')
    
    # 在图上标出最佳点
    opt_inertia = inertias[k_range.index(optimal_k)]
    plt.plot(optimal_k, opt_inertia, marker='*', color='r', markersize=15, label=f'Elbow (K={optimal_k})')
    
    plt.title(f'Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True)
    plt.legend()
    plt.xticks(k_range)
    
    # 保存图片而不是只显示
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_k_elbow_curve.png'))
    # plt.show() # 如果是全自动运行，可以注释掉这一行，避免脚本暂停等待关闭窗口
    
    print(f"\n🤖 Algorithm automatically selected K = {optimal_k}")
    return optimal_k

def predict_multiprocess(kmeans_model, df, norm_stats, n_clusters):
    """多进程结果生成"""
    print(f"\n🚀 Multiprocessing: Predicting & Generating ROIs...")
    
    mu_d, std_d = norm_stats['d_mean'], norm_stats['d_std']
    mu_f, std_f = norm_stats['f_mean'], norm_stats['f_std']
    
    kmeans_centers = kmeans_model.cluster_centers_
    
    centers_display = pd.DataFrame(kmeans_centers, columns=['D_norm_Center', 'f_norm_Center'])
    centers_display['D_Actual'] = centers_display['D_norm_Center'] * std_d + mu_d
    centers_display['f_Actual'] = centers_display['f_norm_Center'] * std_f + mu_f
    print("\n🔬 Cluster Centers (Actual Values):")
    print(centers_display)
    
    tasks = []
    rows = df.to_dict('records')
    for row in rows:
        tasks.append((row, kmeans_centers, mu_d, std_d, mu_f, std_f, n_clusters, OUTPUT_DIR, NORM_IVIM_DIR, ROI_DIR))
        
    volume_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker_predict_and_save, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="  CPU Computing"):
            res = future.result()
            if res:
                volume_results.append(res)
                
    results_df = pd.DataFrame(volume_results)
    if not results_df.empty:
        results_df.to_excel(os.path.join(OUTPUT_DIR, 'cluster_volume_ratios.xlsx'), index=False)
        print(f"✅ Results saved to {OUTPUT_DIR}")
    else:
        print("❌ No results generated.")

# ================= 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"🖥️ CPU: Detected {multiprocessing.cpu_count()} cores. Using {MAX_WORKERS} workers.")
    
    df = pd.read_excel(EXCEL_PATH)
    df[COL_CASE_ID] = df[COL_CASE_ID].astype(str).str.strip()
    
    if not os.path.exists(STATS_PATH):
        print("Stats file not found.")
        return

    with open(STATS_PATH, 'r') as f:
        norm_stats = json.load(f)
        
    # 1. 多进程数据准备
    train_matrix = prepare_data_multiprocess(df, dataset_filter='Training')
    if train_matrix is None: return

    # 2. 确定 K 值 (使用肘部法则)
    # 这一步会弹出图片，并要求您输入 K 值
    optimal_k = determine_optimal_k(train_matrix, k_max=8)
    
    print(f"\n✅ Selected Optimal K: {optimal_k}")
    
    # 3. 训练最终模型
    print(f"\n⚙️ Training Final Model (K={optimal_k})...")
    final_model = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
    final_model.fit(train_matrix)
    
    # 4. 多进程应用模型
    predict_multiprocess(final_model, df, norm_stats, optimal_k)

if __name__ == "__main__":
    main()