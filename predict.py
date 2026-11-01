import joblib
import pandas as pd
import numpy as np
import shap

# 1. Load the trained pipeline artifact
PIPELINE_PATH = "ckd_pipeline.joblib"
pipeline = joblib.load(PIPELINE_PATH)

# Extract preprocessor and model for SHAP local attribution
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['classifier']

# Extract feature names after One-Hot Encoding
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]
encoded_cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols).tolist()
all_feature_names = list(num_cols) + encoded_cat_cols

# Initialize TreeExplainer
explainer = shap.TreeExplainer(model)

def assess_patient_risk(patient_dict: dict) -> dict:
    """
    Takes a dictionary of raw patient features, evaluates risk using the pipeline,
    and returns risk probability, risk category, and individual SHAP attributions.
    """
    # Convert single patient dict to a DataFrame
    input_df = pd.DataFrame([patient_dict])
    
    # 1. Calculate Risk Probability via Pipeline
    prob_ckd = float(pipeline.predict_proba(input_df)[0, 1])
    risk_percentage = round(prob_ckd * 100, 2)
    
    # 2. Map to Clinical Decision Support Risk Tier
    if risk_percentage < 30.0:
        risk_level = "Low Risk"
        decision_support = "Routine monitoring recommended. Biomarkers are within typical baseline ranges."
    elif risk_percentage < 70.0:
        risk_level = "Moderate Risk"
        decision_support = "Borderline clinical markers observed. Recommend lifestyle review and follow-up lab test in 3 months."
    else:
        risk_level = "Elevated Risk"
        decision_support = "Key renal markers exceed expected thresholds. Clinical review by a nephrologist recommended."

    # 3. Compute Local SHAP values for this specific patient
    transformed_input = preprocessor.transform(input_df)
    patient_shap = explainer.shap_values(transformed_input)
    
    # Handle SHAP multi-class output format
    if isinstance(patient_shap, list):
        patient_shap_class1 = patient_shap[1][0]
    elif len(patient_shap.shape) == 3:
        patient_shap_class1 = patient_shap[0, :, 1]
    else:
        patient_shap_class1 = patient_shap[0]
        
    # 4. Extract Top 3 Factors Raising Risk and Top 3 Factors Lowering Risk
    impact_series = pd.Series(patient_shap_class1, index=all_feature_names)
    
    top_risk_drivers = impact_series.sort_values(ascending=False).head(3).to_dict()
    top_protective_factors = impact_series.sort_values(ascending=True).head(3).to_dict()

    return {
        "risk_probability": round(prob_ckd, 4),
        "risk_percentage": f"{risk_percentage}%",
        "risk_level": risk_level,
        "clinical_decision_support": decision_support,
        "top_risk_drivers": {k: round(v, 4) for k, v in top_risk_drivers.items()},
        "top_protective_factors": {k: round(v, 4) for k, v in top_protective_factors.items()}
    }

# ---------------------------------------------------------
# TEST WITH TWO REALISTIC PATIENT SCENARIOS
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== TESTING LOCAL PATIENT INFERENCE ===")
    
    # Patient A: Healthy routine checkup profile
    healthy_patient = {
        'age': 35.0, 'bp': 70.0, 'sg': 1.025, 'al': 0.0, 'su': 0.0,
        'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
        'bgr': 95.0, 'bu': 22.0, 'sc': 0.8, 'sod': 140.0, 'pot': 4.5,
        'hemo': 15.5, 'pcv': 45.0, 'wc': 7200.0, 'rc': 5.2,
        'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
    }

    # Patient B: Elevated clinical markers profile
    at_risk_patient = {
        'age': 62.0, 'bp': 90.0, 'sg': 1.010, 'al': 3.0, 'su': 2.0,
        'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent',
        'bgr': 220.0, 'bu': 85.0, 'sc': 3.4, 'sod': 132.0, 'pot': 5.6,
        'hemo': 8.8, 'pcv': 28.0, 'wc': 11500.0, 'rc': 3.1,
        'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
    }

    print("\n--- PATIENT A (Normal Biomarkers) ---")
    result_a = assess_patient_risk(healthy_patient)
    for k, v in result_a.items():
        print(f"{k}: {v}")

    print("\n--- PATIENT B (Compromised Biomarkers) ---")
    result_b = assess_patient_risk(at_risk_patient)
    for k, v in result_b.items():
        print(f"{k}: {v}")