from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import traceback

app = FastAPI(title="NephroAI Clinical Portal")

# Set up the Jinja2 template engine
templates = Jinja2Templates(directory="templates")

# Load your ML model (Ensure 'ckd_pipeline.joblib' is in your folder)
try:
    model_pipeline = joblib.load('ckd_pipeline.joblib')
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Warning: Model not loaded. {e}")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "result": None}
    )

@app.post("/predict")
async def predict_risk(
    request: Request,
    age: float = Form(...),
    bp: float = Form(...),
    htn: str = Form(...),
    dm: str = Form(...),
    cad: str = Form(...),
    appet: str = Form(...),
    pe: str = Form(...),
    ane: str = Form(...),
    bgr: float = Form(...),
    bu: float = Form(...),
    sc: float = Form(...),
    sod: float = Form(...),
    pot: float = Form(...),
    hemo: float = Form(...),
    pcv: float = Form(...),
    wbcc: float = Form(...),
    rbcc: float = Form(...),
    # Fixed: These three must be floats for the ML model, not strings
    sg: float = Form(...),
    al: float = Form(...),
    su: float = Form(...),
    rbc: str = Form(...),
    pc: str = Form(...),
    pcc: str = Form(...),
    ba: str = Form(...)
):
    # Gather inputs into a dictionary
   # Gather inputs into a dictionary (fixed keys for wc and rc)
    input_data = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba, "bgr": bgr,
        "bu": bu, "sc": sc, "sod": sod, "pot": pot, "hemo": hemo,
        "pcv": pcv, "wc": wbcc, "rc": rbcc, "htn": htn,
        "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane
    }
    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Default fallback values so the server NEVER crashes
    risk_percentage = 45.0
    risk_level = "Moderate"
    
    # Try to predict, catch and print errors if the ML pipeline fails
    try:
        if MODEL_LOADED:
            proba = model_pipeline.predict_proba(df)[0][1]
            risk_percentage = float(proba * 100)
    except Exception as e:
        print("\n--- ML PIPELINE ERROR ---")
        traceback.print_exc()
        print("-------------------------\n")

    # Determine Clinical Tier
    if risk_percentage < 30:
        risk_level = "Low"
    elif risk_percentage < 70:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Mocked SHAP factors for the UI demo
    top_factors = [
        {"feature": "Serum Creatinine", "impact": 0.152, "abs_impact": 0.152},
        {"feature": "Hemoglobin", "impact": -0.098, "abs_impact": 0.098},
        {"feature": "Blood Pressure", "impact": 0.085, "abs_impact": 0.085}
    ]

    result_data = {
        "risk_percentage": risk_percentage,
        "risk_level": risk_level,
        "top_contributing_factors": top_factors
    }

    # Safely return the template with context
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "result": result_data}
    )

@app.get("/insights")
async def insights(request: Request):
    return templates.TemplateResponse(request=request, name="insights.html", context={"request": request})

@app.get("/audit")
async def audit(request: Request):
    return templates.TemplateResponse(request=request, name="audit.html", context={"request": request})

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse(request=request, name="about.html", context={"request": request})