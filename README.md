
# Badminton In-Match Prediction System (Streamlit)

This repository combines your trained badminton model, all generated output files, and a hosted Streamlit dashboard.

## What this app includes

- Live **single-match prediction after Game 1**
- **Batch prediction** from CSV
- Final model **metrics and calibration**
- **EDA visuals**
- **SHAP / LIME explainability**
- A presentation-ready dashboard for GitHub + Streamlit Cloud

## Project structure

```text
badminton_streamlit_system/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── models/
│   ├── badminton_best_model_base.pkl
│   ├── badminton_best_model_calibrated.pkl   # optional, preferred
│   └── model_metadata.json
├── outputs/
│   ├── *.png
│   └── *.csv
└── sample_batch_input.csv
```

## Files you need to copy in

From your training pipeline's `outputs/` folder, copy:

### Into `models/`
- `badminton_best_model_base.pkl`
- `badminton_best_model_calibrated.pkl` (preferred if available)
- `model_metadata.json`

### Into `outputs/`
- `eda_missing_values.png`
- `eda_overview.png`
- `eda_feature_distributions.png`
- `eda_boxplot_g1_score_diff.png`
- `baseline_validation_results.csv`
- `final_test_metrics.csv`
- `roc_curves_test.png`
- `confusion_matrix_final.png`
- `learning_curve_tuned_model.png`
- `calibration_curve_validation.png`
- `feature_importance_model.png`
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`
- `shap_dependence_*.png`
- `shap_local_win_case.png`
- `shap_local_loss_case.png`
- `shap_waterfall_win.png`
- `shap_feature_importance.csv`
- `lime_win_case.png`
- `lime_loss_case.png`

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud

1. Create a new GitHub repository
2. Upload all files from this project
3. Add your model and output files into `models/` and `outputs/`
4. Go to Streamlit Community Cloud
5. Deploy the repo
6. Set the main file to `streamlit_app.py`

## Notes

- The app automatically prefers the calibrated model if present.
- If only the base model exists, it will use that instead.
- If some PNG or CSV files are missing, the app still runs and simply hides those sections.
- Keep model files reasonably small for GitHub. If they are large, use Git LFS.
