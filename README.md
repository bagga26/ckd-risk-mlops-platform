[![CKD MLOps CI Pipeline](https://github.com/bagga26/ckd-risk-mlops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/bagga26/ckd-risk-mlops-platform/actions/workflows/ci.yml)
# Explainable AI & MLOps Healthcare Platform: CKD Risk Assessment

An end-to-end Machine Learning Operations (MLOps) platform providing non-diagnostic clinical risk stratification and explainability for Chronic Kidney Disease (CKD) using the gold-standard UCI benchmark dataset.

---

## 📌 System Overview & Core Capabilities

- **Pipeline Architecture:** Scikit-Learn end-to-end pipeline handling missing value imputation (median for numerical, mode for categorical), feature scaling, and one-hot encoding.
- **Classification Engine:** Tuned Random Forest Classifier achieving 100% Accuracy and ROC-AUC on the benchmark test split.
- **Explainable AI (XAI):** Integrated `shap.TreeExplainer` producing both global biomarker importances and real-time patient-level SHAP attributions (risk drivers vs. protective factors).
- **Backend Service:** Asynchronous FastAPI REST API with Pydantic-enforced clinical boundary checks and microsecond-latency telemetry.
- **Frontend Dashboard:** Integrated responsive web interface served directly via static routing with live asynchronous JSON fetch calls.
- **Production MLOps:** Containerized deployment blueprint (`Dockerfile`), dependency management (`requirements.txt`), and transactional audit logging (`logs/inference_audit.log`).

---

## 📊 Dataset & Biomarker Attributes

Based on the UCI Chronic Kidney Disease benchmark dataset (400 patient instances, 24 clinical features):
- **Demographics & Vitals:** Age, Blood Pressure (`bp`).
- **Urine Chemistry:** Specific Gravity (`sg`), Albumin (`al`), Sugar (`su`), Red Blood Cells (`rbc`), Pus Cells (`pc`), Pus Cell Clumps (`pcc`), Bacteria (`ba`).
- **Blood Chemistry:** Blood Glucose Random (`bgr`), Blood Urea (`bu`), Serum Creatinine (`sc`), Sodium (`sod`), Potassium (`pot`), Hemoglobin (`hemo`), Packed Cell Volume (`pcv`), White Blood Cell Count (`wc`), Red Blood Cell Count (`rc`).
- **Clinical History:** Hypertension (`htn`), Diabetes Mellitus (`dm`), Coronary Artery Disease (`cad`), Appetite (`appet`), Pedal Edema (`pe`), Anemia (`ane`).

---

## 🛠 Project Structure

```text
ML-PROJECT/
├── kidney_disease.csv       # UCI CKD benchmark dataset
├── train.py                 # Preprocessing, cross-model benchmarking & SHAP export
├── predict.py               # Serialized inference and local SHAP factor attribution
├── main.py                  # Production FastAPI service with audit logging
├── ckd_pipeline.joblib      # Serialized end-to-end inference pipeline
├── shap_summary.png         # Global feature importance plot
├── requirements.txt         # Pinned production dependencies
├── Dockerfile               # Multi-stage production container blueprint
├── .dockerignore            # Build exclusion rules
├── logs/
│   └── inference_audit.log  # Real-time inference transactions
└── templates/
    └── index.html           # Clinical decision-support dashboard