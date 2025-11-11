# app/main.py
import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from joblib import load

from src.utils.explain import top3_factors_from_logreg_pipeline

# =========================
# Base & chemins
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent  # .../app
ART_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))

# Variables d'env optionnelles (prioritaires sur l'auto-découverte)
ENV_MODEL_PATH = os.getenv("MODEL_PATH")
ENV_META_PATH = os.getenv("META_PATH")
ENV_EXPL_PATH = os.getenv("EXPLAINER_PATH")

# === Audit logging config ===
LOG_ENABLED = os.getenv("LOG_AUDIT", "true").lower() in {"1", "true", "yes", "on"}
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "requests.jsonl"

# =========================
# Helpers
# =========================
def _glob_latest(patterns: List[str]) -> Optional[Path]:
    """
    Retourne le fichier le plus récent (mtime) parmi une liste de patterns glob.
    Recherche dans ART_DIR uniquement. None si rien ne matche.
    """
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(sorted(ART_DIR.glob(pat)))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _resolve_artifact_paths() -> Dict[str, Optional[Path]]:
    """
    Détermine MODEL/META/EXPL :
    1) si variables d'env fournies => on les utilise telles quelles
    2) sinon, on auto-découvre le dernier artefact par motifs versionnés
    3) sinon, on tente les noms "stables" historiques (model.joblib, etc.)
    """
    # MODEL
    if ENV_MODEL_PATH:
        model = Path(ENV_MODEL_PATH)
    else:
        model = _glob_latest(["*_model_logreg.joblib", "*_model_xgb.joblib"])
        if model is None:
            model = ART_DIR / "model.joblib"

    # META
    if ENV_META_PATH:
        meta = Path(ENV_META_PATH)
    else:
        meta = _glob_latest(["*_metadata_logreg.json", "*_metadata_xgb.json"])
        if meta is None:
            meta = ART_DIR / "metadata.json"

    # EXPL (optionnel, logreg)
    if ENV_EXPL_PATH:
        expl = Path(ENV_EXPL_PATH)
    else:
        expl = _glob_latest(["*_explainer_logreg.joblib"])
        if expl is None:
            expl = ART_DIR / "explainer.joblib"

    return {"model": model, "meta": meta, "expl": expl}

def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _file_status(model: Path, meta: Path, expl: Optional[Path]) -> Dict[str, Any]:
    return {
        "artifacts_dir": str(ART_DIR),
        "model_path": str(model),
        "meta_path": str(meta),
        "explainer_path": str(expl) if expl else None,
        "model_exists": model.exists(),
        "meta_exists": meta.exists(),
        "explainer_exists": (expl.exists() if expl else False),
    }

def get_threshold(meta: Dict[str, Any]) -> float:
    """Toujours utiliser le seuil issu du training (metadata)."""
    try:
        return float(meta.get("threshold_suggested", 0.05))
    except Exception:
        return 0.05


def audit_log(event: Dict[str, Any]) -> None:
    """Écrit une ligne JSON par requête (best effort)."""
    if not LOG_ENABLED:
        return
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass

# =========================
# App & bundle
# =========================
app = FastAPI(
    title="Credit Scoring API (demo)",
    description="Score + décision binaire (seuil configurable) + top-3 facteurs métier.",
    version="1.1.0",
)

bundle: Dict[str, Any] = {
    "pipeline_pred": None,          # joblib pipeline (obligatoire)
    "pipeline_expl": None,          # joblib pipeline (optionnel)
    "meta": {},                     # dict
    "paths": {},                    # pour debug/health
}

@app.on_event("startup")
def load_artifacts() -> None:
    """
    Charge les artefacts au démarrage (et pas à l'import).
    S'appuie sur variables d'env OU auto-découverte des fichiers versionnés.
    """
    paths = _resolve_artifact_paths()
    model_path: Path = paths["model"]
    meta_path: Path = paths["meta"]
    expl_path: Optional[Path] = paths["expl"]

    # Lire meta (si disponible, non bloquant pour charger le modèle)
    meta = _safe_read_json(meta_path)

    # Charger le modèle principal (obligatoire)
    if not model_path.exists():
        raise RuntimeError(
            "Model not found.\n"
            + json.dumps(_file_status(model_path, meta_path, expl_path), indent=2)
        )

    pipeline_pred = load(model_path)

    # Charger l'explainer (optionnel)
    pipeline_expl = None
    if expl_path and expl_path.exists():
        try:
            pipeline_expl = load(expl_path)
        except Exception:
            pipeline_expl = None  # on ignore si illisible/incompatible

    # Hydrater le bundle
    bundle["pipeline_pred"] = pipeline_pred
    bundle["pipeline_expl"] = pipeline_expl
    bundle["meta"] = meta
    bundle["paths"] = _file_status(model_path, meta_path, expl_path)

# =========================
# Schéma d'entrée
# =========================
class CreditApp(BaseModel):
    # On attend un objet JSON avec TOUTES les colonnes brutes du dataset (ordre non important)
    payload: Dict[str, Any]

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    meta = bundle.get("meta") or {}
    return {
        "status": "ok",
        "model_loaded": bundle["pipeline_pred"] is not None,
        "model_version": meta.get("version"),
        "model_name": meta.get("model"),
        "artifacts": bundle.get("paths", {}),
    }

@app.get("/version")
def version():
    return bundle.get("meta") or {}

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
    # Sécurité si le startup a échoué
    if bundle["pipeline_pred"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = bundle["meta"] or {}
    cols = meta.get("features")
    if not cols:
        raise HTTPException(status_code=500, detail="Metadata missing 'features' list")

    # Vérification de complétude
    missing = [c for c in cols if c not in app_in.payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {missing}")

    # Ordonner les colonnes comme à l'entraînement
    x_df = pd.DataFrame([{c: app_in.payload[c] for c in cols}])

    # Prédiction
    proba = float(bundle["pipeline_pred"].predict_proba(x_df)[:, 1][0])
    thr = get_threshold(meta)
    decision = "ACCEPT" if proba < thr else "REFUSE"

    # Explications top-3 (si explainer dispo)
    top_factors = []
    if bundle["pipeline_expl"] is not None:
        try:
            top_factors = top3_factors_from_logreg_pipeline(
                bundle["pipeline_expl"], x_df, meta.get("lexicon", {})
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
            "model_version": meta.get("version"),
            "threshold": thr,
            "probability": proba,
            "decision": decision,
            "top_factors": top_factors,
            # ⚠️ éviter les PII en prod. Dataset pédagogique ici.
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
        "model_version": meta.get("version"),
    }

# =========================
# Fichiers statiques (UI)
# =========================
STATIC_DIR = BASE_DIR / "app" / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
