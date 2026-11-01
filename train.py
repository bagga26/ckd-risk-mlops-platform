import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
file_path = "kidney_disease.csv"
df = pd.read_csv(file_path)

print("=== 1. DATA LOADED ===")
print(f"Dataset Shape: {df.shape}")

# ---------------------------------------------------------
# 2. DROP UNNECESSARY COLUMNS
# ---------------------------------------------------------
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# ---------------------------------------------------------
# 3. CLEAN STRING ARTIFACTS
# ---------------------------------------------------------
# Include both 'object' and 'str' to silence Pandas 3.0+ warnings
text_cols = df.select_dtypes(include=['object', 'str']).columns

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.replace('\t', '')
    df[col] = df[col].replace({'nan': np.nan, '?': np.nan})

# ---------------------------------------------------------
# 4. ENCODE TARGET COLUMN
# ---------------------------------------------------------
# 1 = CKD Risk (Elevated), 0 = Not CKD (Low Risk)
target_mapping = {'ckd': 1, 'notckd': 0}
df['classification'] = df['classification'].map(target_mapping)

# Check for any unmapped target rows
if df['classification'].isnull().any():
    df = df.dropna(subset=['classification'])

# ---------------------------------------------------------
# 5. DEFINE FEATURES AND TARGET
# ---------------------------------------------------------
X = df.drop(columns=['classification'])
y = df['classification']

# Identify numerical and categorical feature names
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()

print(f"\nFeatures identified: {len(num_cols)} Numerical, {len(cat_cols)} Categorical")

# ---------------------------------------------------------
# 6. STRATIFIED TRAIN / TEST SPLIT (80% / 20%)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print("\n=== 2. DATA SPLIT COMPLETED ===")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
print(f"Train class balance: {dict(y_train.value_counts())}")
print(f"Test class balance:  {dict(y_test.value_counts())}")

# ---------------------------------------------------------
# 7. BUILD PRODUCTION-GRADE PREPROCESSING PIPELINE
# ---------------------------------------------------------
# Numerical transformer: Median Imputation + Standard Scaling
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: Mode Imputation + One-Hot Encoding
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine both branches into a master preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Fit on training data ONLY and transform both splits
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n=== 3. PREPROCESSING SUCCESSFUL ===")
print(f"X_train transformed shape: {X_train_processed.shape}")
print(f"X_test transformed shape:  {X_test_processed.shape}")
print(f"Any NaNs remaining in train? {np.isnan(X_train_processed).any()}")
print(f"Any NaNs remaining in test?  {np.isnan(X_test_processed).any()}")
# ---------------------------------------------------------
# 8. TRAIN MULTIPLE MODELS & CLINICAL EVALUATION
# ---------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize candidates
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results = []

print("\n=== 4. TRAINING & EVALUATING CANDIDATE MODELS ===")

for name, model in models.items():
    # 1. Fit on processed training set
    model.fit(X_train_processed, y_train)
    
    # 2. Predict on unseen test set
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    
    # 3. Calculate clinical metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc, 4)
    })

# Convert comparison table to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
# ---------------------------------------------------------
# 9. ASSEMBLE COMPLETE PRODUCTION PIPELINE & EXPORT
# ---------------------------------------------------------
import joblib

print("\n=== 5. BUILDING END-TO-END INFERENCE PIPELINE ===")

# Create the master production pipeline combining preprocessing + selected model
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=42
    ))
])

# Train pipeline on the full training set (raw X_train)
final_pipeline.fit(X_train, y_train)

# Validate on raw test set to ensure pipeline handles end-to-end inference
test_preds = final_pipeline.predict(X_test)
test_probs = final_pipeline.predict_proba(X_test)[:, 1]
final_accuracy = accuracy_score(y_test, test_preds)
final_roc = roc_auc_score(y_test, test_probs)

print(f"End-to-End Pipeline Test Accuracy: {final_accuracy:.4f}")
print(f"End-to-End Pipeline Test ROC-AUC:  {final_roc:.4f}")

# Save the trained pipeline to disk
model_filename = "ckd_pipeline.joblib"
joblib.dump(final_pipeline, model_filename)
print(f"Artifact saved successfully as: '{model_filename}'") 
# ---------------------------------------------------------
# 10. EXPLAINABLE AI WITH SHAP
# ---------------------------------------------------------
import shap
import matplotlib.pyplot as plt

print("\n=== 6. GENERATING SHAP EXPLAINABILITY ===")

# 1. Extract the fitted preprocessor and classifier from the pipeline
fitted_preprocessor = final_pipeline.named_steps['preprocessor']
fitted_model = final_pipeline.named_steps['classifier']

# 2. Get transformed feature names after One-Hot Encoding
encoded_cat_cols = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + encoded_cat_cols

# 3. Transform test features for SHAP evaluation
X_test_transformed = fitted_preprocessor.transform(X_test)

# 4. Initialize TreeExplainer for the Random Forest model
explainer = shap.TreeExplainer(fitted_model)
shap_values = explainer.shap_values(X_test_transformed)

# Handle output format differences across SHAP versions (for binary classification class 1)
if isinstance(shap_values, list):
    # Older SHAP versions return a list of arrays [class_0, class_1]
    shap_vals_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    # Newer versions return an array of shape (samples, features, classes)
    shap_vals_class1 = shap_values[:, :, 1]
else:
    shap_vals_class1 = shap_values

# 5. Global Feature Importance Summary (Top 10 Biomarkers)
mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:10]

print("Top 10 Clinical Risk Contributors (Global SHAP Importance):")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {all_feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

# 6. Local Patient Explanation (Patient 0 from Test Set)
sample_patient_raw = X_test.iloc[[0]]
sample_pred_prob = final_pipeline.predict_proba(sample_patient_raw)[0, 1]
sample_true_label = y_test.iloc[0]

print(f"\nSample Patient 0 Assessment:")
print(f"Predicted CKD Risk Probability: {sample_pred_prob:.4f}")
print(f"Actual Clinical Status:         {'Elevated Risk (CKD)' if sample_true_label == 1 else 'Low Risk (Not CKD)'}")

# Save global SHAP summary plot as an image artifact
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_class1, X_test_transformed, feature_names=all_feature_names, show=False)
plt.title("SHAP Feature Importance (CKD Risk Factors)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()
print("Saved SHAP summary plot as: 'shap_summary.png'")