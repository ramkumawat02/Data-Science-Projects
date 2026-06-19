# ICU Mortality Prediction

An end-to-end machine learning project for predicting in-hospital mortality from ICU patient data derived from MIMIC-IV. The repository covers exploratory analysis, feature engineering, gradient-boosting and LSTM modelling, patient-grouped evaluation, SHAP explainability, saved-model inference, and an interactive Streamlit dashboard.

> This project is intended for research, education, and portfolio demonstration. It is not a clinical decision-support system.

## Project overview

| Component | Implementation |
|---|---|
| Prediction target | In-hospital mortality using `hospital_expire_flag` |
| Data strategy | Processed ICU features with train, validation, and test groups separated by patient ID |
| Models | Logistic Regression, Histogram Gradient Boosting, and a PyTorch LSTM |
| Evaluation | ROC-AUC, PR-AUC, F1, calibration, Brier score, bootstrap confidence intervals, and threshold analysis |
| Explainability | Global feature importance and patient-level SHAP contributions |
| Application | Streamlit dashboard for manual input and sample-patient scoring |

## Saved model results

The following holdout results are recorded in `results/metrics/holdout_model_comparison_lstm_notebook.csv`:

| Model | ROC-AUC | PR-AUC | F1 | Selected threshold |
|---|---:|---:|---:|---:|
| LSTM | **0.795** | 0.448 | 0.381 | 0.75 |
| Logistic Regression | 0.778 | 0.418 | **0.448** | 0.60 |
| Histogram Gradient Boosting | 0.768 | **0.454** | 0.407 | 0.40 |

These results show the trade-off between ranking performance, minority-class precision-recall performance, and threshold-dependent classification quality. Metrics may vary when models are retrained.

## Technical highlights

### Data preparation and validation

- Explores ICU patient characteristics, missingness, distributions, and outcome balance.
- Builds derived clinical and demographic features for modelling.
- Validates schemas, target values, feature types, and model inputs.
- Uses patient-grouped splitting to reduce information leakage between training and evaluation sets.
- Stores processed and inference-safe datasets with accompanying metadata.

### Model development

- Implements reproducible scikit-learn training and evaluation workflows.
- Compares linear, tree-based, and recurrent neural-network approaches.
- Trains a PyTorch LSTM for mortality-risk prediction.
- Selects classification thresholds using validation data instead of optimising on the test set.
- Saves model artefacts for repeatable inspection and inference.

### Explainability

- Uses SHAP to identify the features driving model predictions globally.
- Produces patient-level contribution plots for local interpretation.
- Exports feature-importance tables and written interpretation notes.

![SHAP feature importance](results/figures/icu-shap-summary.png)

### Interactive dashboard

The Streamlit application provides:

- Manual patient-feature entry
- Scoring of sample rows from the processed dataset
- Predicted mortality probability
- Threshold-based risk classification
- Local SHAP contributions for the selected patient

## Repository structure

```text
ICU-Mortality-Prediction/
├── app.py                         Streamlit application
├── config/
│   └── config.yaml                Project configuration
├── data/
│   └── processed/                 Processed and inference-safe datasets
├── models/                        Saved gradient-boosting and LSTM artefacts
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_model_lstm.ipynb
│   └── 05_explainability_shap.ipynb
├── results/
│   ├── figures/                   Evaluation and SHAP visualisations
│   ├── metrics/                   Holdout and cross-validation results
│   └── shap/                      SHAP exports and interpretations
├── src/                           Reusable ML and application modules
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

- Python 3.11 or later
- `pip` and a Python virtual environment
- Raw MIMIC-IV access only if the complete preprocessing pipeline must be regenerated

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The committed scikit-learn artefacts were created with scikit-learn 1.8.0. Using the same version avoids compatibility warnings when loading them.

## Usage

### Launch the dashboard

```bash
streamlit run app.py
```

The processed data and saved inference bundle included in the repository are sufficient to explore the dashboard without downloading raw MIMIC-IV files.

### Inspect saved models

Inspect every supported artefact:

```bash
python src/inspect_models.py
```

Or inspect a specific model:

```bash
python src/inspect_models.py models/gb_model.pkl
python src/inspect_models.py models/gradient_boosting.pkl
python src/inspect_models.py models/gb_shap_inference_bundle.pkl
python src/inspect_models.py models/lstm_model.pt
```

The inspector reports model types, feature names, key hyperparameters, bundle metadata, and checkpoint structure.

### Run a sample prediction

```bash
python src/predict_gb.py --artifact models/gb_model.pkl --row 0
```

This loads a processed sample, prints its feature values, and returns the predicted probability and classification.

### Train the gradient-boosting model

```bash
python -m src.train
```

The training workflow reads `data/processed/icu_features.csv` and writes the fitted model to `models/gradient_boosting.pkl`.

## Evaluation artefacts

| Artefact | Purpose |
|---|---|
| `results/metrics/holdout_report.json` | Test metrics, bootstrap confidence intervals, split details, and selected threshold |
| `results/metrics/holdout_model_comparison_lstm_notebook.csv` | LSTM and classical-model comparison |
| `results/metrics/cv_model_comparison.csv` | Cross-validation model comparison |
| `results/figures/roc_curve.png` | ROC performance visualisation |
| `results/figures/cv_calibration_comparison.png` | Calibration comparison across models |
| `results/shap/shap_interpretation.md` | Written interpretation of SHAP results |

## Data availability and responsible use

- Raw MIMIC-IV files are excluded from Git tracking and require authorised access from the original data provider.
- The repository includes derived features for demonstration and application workflows.
- The prediction target represents **in-hospital mortality**, not ICU-only mortality.
- Model outputs must not be used for diagnosis, treatment, triage, or other clinical decisions.
- Performance on the included split does not establish generalisation to another hospital, population, or clinical setting.

## Development note

GitHub Copilot assisted with documentation, type hints, error handling, and refactoring. The machine-learning design, clinical-domain logic, feature-engineering strategy, and project structure are original work.
