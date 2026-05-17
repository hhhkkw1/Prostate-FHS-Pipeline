# Main Scripts for Image Processing and Statistical Analysis

This folder contains the main scripts for image processing and statistical analysis.

## Script Mapping

1. `01_preprocess_resample_normalize.py`  
   Resample ROI and IVIM volumes, then compute training-based normalization stats and export normalized IVIM maps.

2. `02_kmeans_habitat_clustering.py`  
   Train KMeans on normalized training voxels, export clustered ROI maps and cluster volume summaries.

3. `03_extract_habitat_features.py`  
   Extract habitat-level morphology, spatial, and first-order features from clustered ROIs.

4. `04_lasso_feature_selection.R`  
   Perform LASSO feature selection and compute `FHS` (Functional Habitat Score).

5. `05_compute_capra_score.py`  
   Compute `CAPRA_score` from standardized clinical variables and append it to the LASSO output table.

6. `06_fit_models_and_predict.py`  
   Fit logistic models (CAPRA / Clinical / ImageModel / Habitat / Combined) and export predicted probabilities.

7. `07_evaluate_models_and_plot_roc.R`  
   Evaluate model performance (AUC, CI, sensitivity/specificity, NRI, IDI) and generate ROC plots.

## Unified Configuration

All paths, key column names, dataset split values, and major parameters are centralized in:

- `config/pipeline_config.json`

Please update this file first before running.

## Run Order

Run scripts sequentially:

1. `python 01_preprocess_resample_normalize.py`
2. `python 02_kmeans_habitat_clustering.py`
3. `python 03_extract_habitat_features.py`
4. `Rscript 04_lasso_feature_selection.R`
5. `python 05_compute_capra_score.py`
6. `python 06_fit_models_and_predict.py`
7. `Rscript 07_evaluate_models_and_plot_roc.R`

## I/O Linking

The pipeline is chained through config-defined files:

- Step 3 output: `feature_output_excel`
- Step 4 output: `lasso_output_excel`
- Step 5 output: `capra_output_excel`
- Step 6 output: `prediction_output_excel`
- Step 7 input: `prediction_output_excel`

This ensures consistent variable names and file handoff across all steps.

## Key Standardized Variables

- Case ID: `Study_ID` (fallback candidates can be configured)
- Split column: `Cohort`
- Outcome: `AP_Status`
- Clinical columns:
  - `Age`
  - `Preoperative_PSA`
  - `T_stage`
  - `PI-RADS`
  - `pct_pos`
  - `GG_BP`
  - `CAPRA_score`
- Habitat score from LASSO / model input: `FHS` (Functional Habitat Score)
- Prediction columns:
  - `pred_prob_CAPRA`
  - `pred_prob_Clinical`
  - `pred_prob_ImageModel`
  - `pred_prob_Habitat`
  - `pred_prob_Combined`

