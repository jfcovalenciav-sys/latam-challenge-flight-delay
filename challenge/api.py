from __future__ import annotations

import os
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from starlette.status import HTTP_400_BAD_REQUEST

from challenge.model import DelayModel

app = FastAPI(title="Flight Delay API", version="1.0.0")

# --- 1) Handler: convertir 422 -> 400 y serializar correctamente los errores ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errs = exc.errors()
    # exc.errors() puede traer ctx={"error": ValueError(...)} que NO es serializable
    for e in errs:
        ctx = e.get("ctx")
        if ctx and "error" in ctx and isinstance(ctx["error"], Exception):
            ctx["error"] = str(ctx["error"])
    return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"detail": errs})

# --- 2) Rutas robustas para el dataset, relativo a este archivo ---
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.csv"))

_model: DelayModel | None = None
_known_airlines: set[str] | None = None

# --- 3) Esquemas de entrada/salida ---
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @field_validator("TIPOVUELO")
    @classmethod
    def _tipo_ok(cls, v: str) -> str:
        if v not in {"I", "N"}:
            raise ValueError("TIPOVUELO must be 'I' or 'N'")
        return v

    @field_validator("MES")
    @classmethod
    def _mes_ok(cls, v: int) -> int:
        if not (1 <= v <= 12):
            raise ValueError("MES must be between 1 and 12")
        return v

class PredictRequest(BaseModel):
    flights: List[Flight]

class PredictResponse(BaseModel):
    predict: List[int]

# --- 4) Carga del modelo en startup (y con fallback desde /predict por si no corrió) ---
@app.on_event("startup")
def _load_model():
    global _model, _known_airlines
    m = DelayModel()
    m.fit_from_csv(DATA_PATH)  # lanza FileNotFoundError si no existe
    _model = m
    _known_airlines = set(m.known_airlines_ or [])

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global _model, _known_airlines

    # Fallback: si por alguna razón no corrió el startup en los tests
    if _model is None or _known_airlines is None:
        _load_model()

    # Validar aerolínea conocida (el test solo chequea status=400 cuando algo está mal)
    unknown = sorted({f.OPERA for f in req.flights if f.OPERA not in _known_airlines})
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown OPERA: {unknown}")

    # Pydantic v2 usa .model_dump(); en v1 .dict(). Usamos ambos por compatibilidad
    rows = [(f.model_dump() if hasattr(f, "model_dump") else f.dict()) for f in req.flights]
    df = pd.DataFrame(rows)

    # Preprocesar y predecir con tu DelayModel (ya alinea a las 10 columnas)
    X = _model.preprocess(df)
    preds = _model.predict(X)

    return PredictResponse(predict=preds)
