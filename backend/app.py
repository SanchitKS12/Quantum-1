from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import requests
import io

from quantum_malware_hunter import load_detector, dataframe_to_array

MODEL_PATH = "detector.joblib"
detector = load_detector(MODEL_PATH)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class MalwareScanner:
    def __init__(self, model):
        self.model = model

    def scan_manual(self, values):
        fused, p_cl, p_q = self.model.predict_proba([values])
        result = {
            "action": "alert" if fused[0] >= 0.5 else "ok",
            "prob": float(fused[0]) * 100
        }
        return result

    def scan_csv_file(self, file_bytes):
        df = pd.read_csv(io.BytesIO(file_bytes))
        X = dataframe_to_array(df)
        fused, _, _ = self.model.predict_proba(X)
        first = fused[0]
        return {
            "rows": len(df),
            "first_action": "alert" if first >= 0.5 else "ok",
            "first_prob": float(first) * 100
        }

    def scan_csv_url(self, url):
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.text))
        X = dataframe_to_array(df)
        fused, _, _ = self.model.predict_proba(X)
        first = fused[0]
        return {
            "rows": len(df),
            "first_action": "alert" if first >= 0.5 else "ok",
            "first_prob": float(first) * 100
        }


scanner = MalwareScanner(detector)


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
    values = [f1, f2, f3, f4, f5, f6]
    result = scanner.scan_manual(values)
    return templates.TemplateResponse("index.html",
                {"request": request, "manual_result": result})


@app.post("/scan/file", response_class=HTMLResponse)
async def scan_file(request: Request, file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = scanner.scan_csv_file(file_bytes)
    return templates.TemplateResponse("index.html",
                {"request": request, "file_result": result})


@app.post("/scan/url", response_class=HTMLResponse)
async def scan_url(request: Request, url: str = Form(...)):
    result = scanner.scan_csv_url(url)
    return templates.TemplateResponse("index.html",
                {"request": request, "url_result": result})
