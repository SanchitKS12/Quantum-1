from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io, requests, pandas as pd, numpy as np, joblib

from quantum_malware_hunter import HybridDetector  # CORRECT import

MODEL_PATH = "detector.joblib"
detector: HybridDetector = joblib.load(MODEL_PATH)  # load saved model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


EXPECTED_COLS = [
    "file_ops_rate", "failed_access_ratio", "new_procs_created",
    "network_conn_count", "unusual_file_types", "entropy_change"
]

def dataframe_to_array(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    return df[EXPECTED_COLS].values


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/scan/manual", response_class=HTMLResponse)
async def scan_manual(
    request: Request,
    f1: float = Form(...),
    f2: float = Form(...),
    f3: float = Form(...),
    f4: float = Form(...),
    f5: float = Form(...),
    f6: float = Form(...)
):
    values = np.array([[f1, f2, f3, f4, f5, f6]])
    fused, _, _ = detector.predict_proba(values)
    result = {
        "action": "alert" if fused[0] >= 0.5 else "ok",
        "prob": round(float(fused[0]) * 100, 2)
    }
    return templates.TemplateResponse("index.html", {"request": request, "manual_result": result})


@app.post("/scan/file", response_class=HTMLResponse)
async def scan_file(request: Request, file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=413, detail="File too large")
    df = pd.read_csv(io.BytesIO(file_bytes))
    X = dataframe_to_array(df)
    fused, _, _ = detector.predict_proba(X)
    result = {
        "rows": len(df),
        "first_action": "alert" if fused[0] >= 0.5 else "ok",
        "first_prob": round(float(fused[0]) * 100, 2)
    }
    return templates.TemplateResponse("index.html", {"request": request, "file_result": result})


@app.post("/scan/url", response_class=HTMLResponse)
async def scan_url(request: Request, url: str = Form(...)):
    try:
        csv_data = requests.get(url).text
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreachable CSV URL")

    X = dataframe_to_array(df)
    fused, _, _ = detector.predict_proba(X)
    result = {
        "rows": len(df),
        "first_action": "alert" if fused[0] >= 0.5 else "ok",
        "first_prob": round(float(fused[0]) * 100, 2)
    }
    return templates.TemplateResponse("index.html", {"request": request, "url_result": result})
