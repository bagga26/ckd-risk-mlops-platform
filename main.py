import os
import time
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any

from predict import assess_patient_risk

# ---------------------------------------------------------
# 1. MLOps Logging Setup
# ---------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "inference_audit.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CKD_MLOps")

# ---------------------------------------------------------
# 2. FastAPI Application Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Chronic Kidney Disease Risk Assessment API",
    description="Explainable Healthcare Decision Support System with Real-Time Monitoring",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 3. Pydantic Input Schema with Physiological Boundary Checks
# ---------------------------------------------------------
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years (1 - 120)")
    bp: float = Field(..., ge=40, le=220, description="Blood Pressure in mm/Hg (40 - 220)")
    sg: float = Field(..., ge=1.000, le=1.035, description="Specific Gravity (1.000 - 1.035)")
    al: float = Field(..., ge=0, le=5, description="Albumin scale 0 to 5")
    su: float = Field(..., ge=0, le=5, description="Sugar scale 0 to 5")
    rbc: str = Field(..., pattern="^(normal|abnormal)$")
    pc: str = Field(..., pattern="^(normal|abnormal)$")
    pcc: str = Field(..., pattern="^(present|notpresent)$")
    ba: str = Field(..., pattern="^(present|notpresent)$")
    bgr: float = Field(..., ge=20, le=600, description="Blood Glucose Random (20 - 600 mg/dl)")
    bu: float = Field(..., ge=1, le=450, description="Blood Urea (1 - 450 mg/dl)")
    sc: float = Field(..., ge=0.1, le=50.0, description="Serum Creatinine (0.1 - 50.0 mg/dl)")
    sod: float = Field(..., ge=80, le=190, description="Sodium (80 - 190 mEq/L)")
    pot: float = Field(..., ge=1.5, le=15.0, description="Potassium (1.5 - 15.0 mEq/L)")
    hemo: float = Field(..., ge=2.0, le=22.0, description="Hemoglobin (2.0 - 22.0 g/dL)")
    pcv: float = Field(..., ge=10, le=65, description="Packed Cell Volume (10 - 65%)")
    wc: float = Field(..., ge=1000, le=35000, description="White Blood Cell Count")
    rc: float = Field(..., ge=1.0, le=10.0, description="Red Blood Cell Count")
    htn: str = Field(..., pattern="^(yes|no)$")
    dm: str = Field(..., pattern="^(yes|no)$")
    cad: str = Field(..., pattern="^(yes|no)$")
    appet: str = Field(..., pattern="^(good|poor)$")
    pe: str = Field(..., pattern="^(yes|no)$")
    ane: str = Field(..., pattern="^(yes|no)$")

# ---------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: templates/index.html not found.</h3>"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CKD Decision Support API",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": True
    }

@app.post("/predict")
def predict_ckd_risk(patient: PatientData):
    start_time = time.perf_counter()
    patient_dict = patient.model_dump()
    
    try:
        # Run ML Pipeline & SHAP Explainer
        assessment = assess_patient_risk(patient_dict)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        assessment["inference_latency_ms"] = latency_ms

        # Structured MLOps Audit Logging
        log_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "patient_age": patient_dict["age"],
            "risk_percentage": assessment["risk_percentage"],
            "risk_level": assessment["risk_level"],
            "latency_ms": latency_ms,
            "status": "SUCCESS"
        }
        logger.info(f"AUDIT_RECORD: {json.dumps(log_payload)}")

        return assessment

    except Exception as e:
        logger.error(f"INFERENCE_FAILURE: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")

# ---------------------------------------------------------
# 5. Local Execution Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)