# src/train/train.py
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from joblib import dump

RAW = Path("data/raw/german_credit.csv")
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)

LEXICON = {
    "duration": "Durée du crédit (mois)",
    "credit_amount": "Montant du crédit",
    "age": "Âge",
    "credit_history": "Historique de crédit",
    "checking_status": "Statut compte courant",
    "savings_status": "Épargne",
    "employment": "Ancienneté emploi",
    "housing": "Logement",
    "job": "Emploi (cat.)",
    "purpose": "Objet du crédit",
    "checking_status": "Statut compte courant",
    "foreign_worker": "Travailleur étranger",
    "installment_commitment": "Taux de mensualité / revenu",
    "existing_credits": "Nb de crédits existants",
    "residence_since": "Ancienneté de résidence",
    "personal_status": "Statut personnel",
    "other_parties": "Autres garants",
    "other_payment_plans": "Autres plans de paiement",
    "property_magnitude": "Patrimoine",
    "num_dependents": "Nombre de personnes à charge",
    "own_telephone": "Téléphone",
    "foreign_worker": "Travailleur étranger",
}

def load_data() -> pd.DataFrame:
    return pd.read_csv(RAW)

def split(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_preprocessor(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return pre, num_cols, cat_cols

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    pre, num_cols, cat_cols = build_preprocessor(X_train)

    # --- Modèle principal (calibré) pour la PREDICTION ---
    base = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf_cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
    pipe_pred = Pipeline([("pre", pre), ("clf", clf_cal)])
    pipe_pred.fit(X_train, y_train)
    proba = pipe_pred.predict_proba(X_test)[:, 1]
    threshold = 0.05
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, proba)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)

    # --- Modèle EXPLAINER (logistique non calibrée) pour les contributions ---
    clf_expl = LogisticRegression(max_iter=1000, solver="lbfgs")
    pipe_expl = Pipeline([("pre", pre), ("clf", clf_expl)])
    pipe_expl.fit(X_train, y_train)

    # Sauvegardes
    from joblib import dump
    dump(pipe_pred, ART / "model.joblib")          # utilisé pour prédire la probabilité
    dump(pipe_expl, ART / "explainer.joblib")      # utilisé pour top-3 facteurs

    meta = {
        "model": "LogisticRegression (calibrated) + explainer LogisticRegression",
        "version": "1.1.0",
        "features": list(X_train.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "metrics": {"auc": float(auc), "precision": float(prec), "recall": float(rec), "f1": float(f1)},
        "threshold_suggested": threshold,
        "lexicon": LEXICON,
    }
    (ART / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Training done ✅  AUC={auc:.3f}  (artifacts in artifacts/)")
    
if __name__ == "__main__":
    main()
