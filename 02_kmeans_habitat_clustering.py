import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from pipeline_utils import detect_case_id_col, ensure_dir, load_config, normalize_case_id


def canonical_case_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def worker_load_and_sample(task: Tuple) -> Optional[Dict[str, np.ndarray]]:
    case_id, n_target, seed, d_path, f_path, roi_path = task
    if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(roi_path)):
        return None

    try:
        d_norm = nib.load(d_path).get_fdata()
        f_norm = nib.load(f_path).get_fdata()
        roi = nib.load(roi_path).get_fdata()

        mask = roi > 0.5
        n_actual = int(np.sum(mask))
        if n_actual < 10:
            return None

        d_pixels = d_norm[mask].reshape(-1)
        f_pixels = f_norm[mask].reshape(-1)

        if n_actual <= n_target:
            idx = np.arange(n_actual)
        else:
            rng = np.random.default_rng(seed + (hash(case_id) % 10000))
            all_idx = np.arange(n_actual)
            rng.shuffle(all_idx)
            step = n_actual / n_target
            idx = all_idx[np.round(np.arange(0, n_actual, step)).astype(int)]
            idx = idx[:n_target]

        return {
            "D_norm": d_pixels[idx],
            "f_norm": f_pixels[idx],
        }
    except Exception:
        return None


def worker_predict_and_save(task: Tuple) -> Optional[Dict]:
    (
        row_dict,
        centers,
        n_clusters,
        norm_ivim_dir,
        roi_resampled_dir,
        cluster_output_dir,
    ) = task
    case_id = canonical_case_id(row_dict["CaseID"])
    split_value = row_dict["Dataset_Type"]

    d_path = os.path.join(norm_ivim_dir, f"{case_id}_ivim_d_norm.nii.gz")
    f_path = os.path.join(norm_ivim_dir, f"{case_id}_ivim_f_norm.nii.gz")
    roi_path = os.path.join(roi_resampled_dir, f"{case_id}.nii.gz")
    if not (os.path.exists(d_path) and os.path.exists(f_path) and os.path.exists(roi_path)):
        return None

    try:
        d_img = nib.load(d_path)
        d_data = d_img.get_fdata()
        f_data = nib.load(f_path).get_fdata()
        roi_data = nib.load(roi_path).get_fdata()

        mask = roi_data > 0.5
        if np.sum(mask) < 10:
            return None

        x_case = np.column_stack([d_data[mask], f_data[mask]])
        distances = np.linalg.norm(x_case[:, np.newaxis, :] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        out_map = np.zeros_like(d_data, dtype=np.uint8)
        out_map[mask] = labels + 1

        out_path = os.path.join(cluster_output_dir, f"{case_id}_cluster_ROI.nii.gz")
        nib.save(nib.Nifti1Image(out_map, d_img.affine, d_img.header), out_path)

        voxel_volume_ml = abs(np.prod(d_img.header.get_zooms()[:3])) / 1000.0
        total_voxels = len(labels)
        total_volume_ml = total_voxels * voxel_volume_ml

        counts = pd.Series(labels).value_counts().sort_index()
        row = {
            "CaseID": case_id,
            "Dataset_Type": split_value,
            "Total_Volume_ml": total_volume_ml,
        }
        for i in range(n_clusters):
            c = int(counts.get(i, 0))
            row[f"Cluster_{i}_Ratio"] = c / total_voxels
            row[f"Cluster_{i}_Volume_ml"] = c * voxel_volume_ml
        return row
    except Exception:
        return None


def build_training_matrix(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    paths = cfg["paths"]
    split_col = cfg["columns"]["split_col"]
    training_value = cfg["dataset_values"]["training"]
    km_cfg = cfg["kmeans"]

    norm_ivim_dir = paths["ivim_normalized_dir"]
    roi_resampled_dir = paths["roi_resampled_dir"]
    n_target = int(km_cfg["sample_per_case"])
    seed = int(km_cfg["random_seed"])
    max_workers = int(km_cfg["max_workers"])

    train_df = df[df[split_col].astype(str).str.strip() == training_value].copy()
    tasks = []
    for _, row in train_df.iterrows():
        case_id = canonical_case_id(row["CaseID"])
        d_path = os.path.join(norm_ivim_dir, f"{case_id}_ivim_d_norm.nii.gz")
        f_path = os.path.join(norm_ivim_dir, f"{case_id}_ivim_f_norm.nii.gz")
        roi_path = os.path.join(roi_resampled_dir, f"{case_id}.nii.gz")
        tasks.append((case_id, n_target, seed, d_path, f_path, roi_path))

    chunks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_load_and_sample, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading training pixels"):
            res = future.result()
            if res is not None:
                chunks.append(pd.DataFrame(res))

    if not chunks:
        raise RuntimeError("No training pixels were loaded for KMeans.")
    pixel_df = pd.concat(chunks, ignore_index=True)
    return pixel_df[["D_norm", "f_norm"]].to_numpy()


def find_elbow_by_max_distance(k_values: np.ndarray, inertias: np.ndarray) -> int:
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    i_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    line_vec = np.array([1.0, -1.0])
    line_norm = np.linalg.norm(line_vec)

    distances = []
    for i in range(len(k_values)):
        point_vec = np.array([k_norm[i], i_norm[i]]) - np.array([0.0, 1.0])
        d = abs(np.cross(line_vec, point_vec)) / line_norm
        distances.append(d)
    return int(k_values[int(np.argmax(distances))])


def determine_k_and_plot(training_matrix: np.ndarray, cfg: dict) -> Tuple[int, pd.DataFrame]:
    km_cfg = cfg["kmeans"]
    plot_dir = cfg["paths"]["plot_dir"]
    ensure_dir(plot_dir)

    k_min = int(km_cfg["k_min"])
    k_max = int(km_cfg["k_max"])
    seed = int(km_cfg["random_seed"])

    k_range = list(range(k_min, k_max + 1))
    inertias = []
    for k in tqdm(k_range, desc="Fitting KMeans for elbow curve"):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        model.fit(training_matrix)
        inertias.append(model.inertia_)

    k_arr = np.array(k_range)
    inertia_arr = np.array(inertias)
    k_auto = find_elbow_by_max_distance(k_arr, inertia_arr)

    fixed_k = km_cfg.get("fixed_k", None)
    if fixed_k is not None and int(fixed_k) >= 2:
        k_selected = int(fixed_k)
    else:
        k_selected = k_auto

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker="o", lw=1.5, label="Inertia (WCSS)")
    plt.scatter([k_auto], [inertias[k_range.index(k_auto)]], c="red", s=80, label=f"Auto elbow K={k_auto}")
    if k_selected != k_auto:
        plt.scatter([k_selected], [inertias[k_range.index(k_selected)]], c="black", s=80, label=f"Selected K={k_selected}")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "kmeans_elbow_curve.png"), dpi=300)
    plt.close()

    summary = pd.DataFrame({"K": k_range, "Inertia": inertias})
    summary["K_auto"] = k_auto
    summary["K_selected"] = k_selected
    return k_selected, summary


def apply_model_to_all(final_model: KMeans, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    paths = cfg["paths"]
    km_cfg = cfg["kmeans"]
    max_workers = int(km_cfg["max_workers"])

    norm_ivim_dir = paths["ivim_normalized_dir"]
    roi_resampled_dir = paths["roi_resampled_dir"]
    cluster_output_dir = paths["cluster_output_dir"]
    ensure_dir(cluster_output_dir)

    centers = final_model.cluster_centers_
    n_clusters = int(final_model.n_clusters)

    tasks = []
    for row in df.to_dict("records"):
        tasks.append(
            (
                row,
                centers,
                n_clusters,
                norm_ivim_dir,
                roi_resampled_dir,
                cluster_output_dir,
            )
        )

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_predict_and_save, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting all cases"):
            res = future.result()
            if res is not None:
                rows.append(res)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train KMeans habitats and export cluster ROI maps.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"CPU cores detected: {multiprocessing.cpu_count()}")

    df = pd.read_excel(cfg["paths"]["input_excel"])
    case_id_col = detect_case_id_col(df, cfg["columns"]["case_id_candidates"])
    df = normalize_case_id(df, case_id_col, out_col="CaseID")

    split_col = cfg["columns"]["split_col"]
    if split_col not in df.columns:
        raise ValueError(f"Missing split column: {split_col}")

    train_matrix = build_training_matrix(df, cfg)
    k_selected, elbow_df = determine_k_and_plot(train_matrix, cfg)
    print(f"Selected K: {k_selected}")

    seed = int(cfg["kmeans"]["random_seed"])
    final_model = KMeans(n_clusters=k_selected, random_state=seed, n_init=10)
    final_model.fit(train_matrix)

    results_df = apply_model_to_all(final_model, df, cfg)
    out_dir = cfg["paths"]["cluster_output_dir"]
    ensure_dir(out_dir)

    ratios_path = os.path.join(out_dir, "cluster_volume_ratios.xlsx")
    elbow_path = os.path.join(out_dir, "kmeans_elbow_table.xlsx")
    centers_path = os.path.join(out_dir, "kmeans_centers.xlsx")

    results_df.to_excel(ratios_path, index=False)
    elbow_df.to_excel(elbow_path, index=False)

    centers = pd.DataFrame(final_model.cluster_centers_, columns=["D_norm_center", "f_norm_center"])
    centers.insert(0, "Cluster", np.arange(centers.shape[0]))
    centers.to_excel(centers_path, index=False)

    print("Saved:", ratios_path)
    print("Saved:", elbow_path)
    print("Saved:", centers_path)
    print("Pipeline step completed: kmeans_habitat_clustering")


if __name__ == "__main__":
    main()
