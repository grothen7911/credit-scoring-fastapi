# app/main.py
import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from joblib import load

from src.utils.explain import top3_factors_from_logreg_pipeline

# =========================
# Config & chemins
# =========================
ART = Path("artifacts")
META = ART / "metadata.json"
MODEL = ART / "model.joblib"        # pipeline calibré (prédiction)
EXPL = ART / "explainer.joblib"     # pipeline logreg (explication)

if not MODEL.exists():
    raise RuntimeError("Model not found. Run the training script first.")

# === Audit logging config ===
LOG_ENABLED = os.getenv("LOG_AUDIT", "true").lower() in {"1", "true", "yes", "on"}
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "requests.jsonl"

# =========================
# App & chargement modèles
# =========================
app = FastAPI(
    title="Credit Scoring API (demo)",
    description="Score + décision binaire (seuil configurable) + top-3 facteurs métier.",
    version="1.1.0",
)

bundle = {
    "pipeline_pred": load(MODEL),                              # prédiction proba
    "pipeline_expl": load(EXPL) if EXPL.exists() else None,   # explication top-3 (si dispo)
    "meta": json.loads(META.read_text(encoding="utf-8")),
}

# =========================
# Schémas d'entrée
# =========================
class CreditApp(BaseModel):
    # On attend un objet JSON avec TOUTES les colonnes brutes du dataset (ordre non important)
    payload: Dict[str, Any]

# =========================
# Utils
# =========================
def get_threshold() -> float:
    """Seuil configurable via env THRESHOLD, sinon valeur suggérée dans metadata."""
    try:
        return float(os.getenv("THRESHOLD", bundle["meta"].get("threshold_suggested", 0.05)))
    except Exception:
        return 0.05

def audit_log(event: Dict[str, Any]) -> None:
    """
    Écrit une ligne JSON par requête dans logs/requests.jsonl.
    Ne casse jamais la réponse si le log échoue.
    """
    if not LOG_ENABLED:
        return
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["meta"].get("model"), "version": bundle["meta"].get("version")}

@app.get("/version")
def version():
    return bundle["meta"]

@app.get("/debug/log-config")
def debug_log_config():
    return {
        "LOG_ENABLED": LOG_ENABLED,
        "LOG_DIR": str(LOG_DIR),
        "LOG_FILE": str(LOG_FILE),
        "exists_dir": LOG_DIR.exists(),
        "exists_file": LOG_FILE.exists(),
    }

@app.post("/score")
def score(app_in: CreditApp, request: Request):
    cols = bundle["meta"]["features"]

    # Vérification de complétude
    missing = [c for c in cols if c not in app_in.payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {missing}")

    # Ordonner les colonnes comme à l'entraînement
    x_df = pd.DataFrame([{c: app_in.payload[c] for c in cols}])

    # Prédiction
    proba = float(bundle["pipeline_pred"].predict_proba(x_df)[:, 1][0])
    thr = get_threshold()
    decision = "ACCEPT" if proba < thr else "REFUSE"

    # Explications top-3 (si explainer dispo)
    top_factors = []
    if bundle["pipeline_expl"] is not None:
        try:
            top_factors = top3_factors_from_logreg_pipeline(
                bundle["pipeline_expl"], x_df, bundle["meta"].get("lexicon", {})
            )
        except Exception:
            top_factors = []

    # Audit log (JSONL)
    try:
        req_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat() + "Z"
        client_ip = request.client.host if request.client else None
        audit_event = {
            "request_id": req_id,
            "timestamp_utc": now,
            "client_ip": client_ip,
            "route": "/score",
            "model_version": bundle["meta"].get("version"),
            "threshold": thr,
            "probability": proba,
            "decision": decision,
            "top_factors": top_factors,
            # ⚠️ Évite de logguer des PII en prod. Dataset pédagogique ici.
            "payload": {c: app_in.payload.get(c, None) for c in cols},
            "user_agent": request.headers.get("user-agent"),
        }
        audit_log(audit_event)
    except Exception:
        pass

    return {
        "probability": proba,
        "decision": decision,
        "threshold": thr,
        "top_factors": top_factors,
        "model_version": bundle["meta"].get("version"),
    }

# =========================
# Fichiers statiques (UI)
# =========================
# Sert app/static/index.html à la racine /
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
