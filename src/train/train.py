# src/train/train.py
# -*- coding: utf-8 -*-
"""
Entraînement + tuning (CV) pour Credit Scoring (LogReg / XGBoost)
- HalvingRandomSearchCV (un seul scoring string, compatible sklearn)
- Métriques finales : ROC-AUC + PR-AUC (Average Precision)
- Pipelines sparse (OneHotEncoder sparse + StandardScaler(with_mean=False))
- LogisticRegression(saga) avec L1/L2/ElasticNet
- Espace XGBoost enrichi
- Calibration (isotonic si données suffisantes, sinon sigmoid)
- Sélection de seuil sur validation interne (pas de fuite du test)
- Export cv_results_ et artefacts versionnés (run_id)
- MLflow : tracking (params, metrics), artefacts (csv/json), modèle (signature)
"""

import os
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime
import warnings

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold

# Activer les recherches "successives" (Halving*)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# xgboost optionnel
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)

# Dossiers
ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

# Dataset attendu
DATA_CSV = Path("data/raw/german_credit.csv")
RANDOM_STATE_DEFAULT = 42


def load_data() -> Tuple[pd.DataFrame, str, list, list]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset introuvable: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    target = "target" if "target" in df.columns else [c for c in df.columns if c.lower() in {"class", "target"}][0]
    if target != "target":
        df = df.rename(columns={target: "target"})
    target = "target"

    cols = [c for c in df.columns if c != target]
    num_cols = ["duration", "credit_amount", "installment_commitment",
                "residence_since", "age", "existing_credits", "num_dependents"]
    cat_cols = [c for c in cols if c not in num_cols]

    # gestion ultra simple des NA
    if df[cat_cols].isna().any().any() or df[num_cols].isna().any().any():
        df = df.dropna().reset_index(drop=True)

    return df, target, num_cols, cat_cols


def make_ohe_sparse(cat_cols):
    """Compatibilité sklearn: >=1.2 => sparse_output, sinon sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preproc(num_cols, cat_cols) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", make_ohe_sparse(cat_cols), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=1.0,
    )


def build_pipe_and_space(model_type: str, num_cols, cat_cols, random_state: int, n_jobs: int):
    pre = build_preproc(num_cols, cat_cols)

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            n_jobs=n_jobs,
            random_state=random_state,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        space = {
            "clf__penalty": ["l2", "l1", "elasticnet"],
            "clf__l1_ratio": np.linspace(0.0, 1.0, 11),   # utilisé si elasticnet
            "clf__C": np.logspace(-3, 2, 40),
        }
        return pipe, space

    if model_type == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost non installé. Faites `pip install xgboost`.")
        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="logloss",
            tree_method="hist",
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        space = {
            "clf__n_estimators": np.arange(400, 1600, 100),
            "clf__learning_rate": np.linspace(0.02, 0.2, 20),
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__min_child_weight": [1, 2, 5, 10],
            "clf__gamma": np.linspace(0.0, 5.0, 11),
            "clf__reg_lambda": np.linspace(0.0, 4.0, 21),
            "clf__reg_alpha": np.linspace(0.0, 3.0, 16),
            "clf__subsample": np.linspace(0.6, 1.0, 9),
            "clf__colsample_bytree": np.linspace(0.6, 1.0, 9),
            "clf__max_bin": [128, 256, 512],
            "clf__grow_policy": ["depthwise", "lossguide"],
            # "clf__scale_pos_weight" sera ajouté dynamiquement
        }
        return pipe, space

    raise ValueError("model_type doit être 'logreg' ou 'xgb'.")


def eval_metrics(y_true, proba, thr: float) -> Dict[str, Any]:
    pred = (proba >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(thr),
    }


def to_json_safe(obj):
    """Convertit récursivement les types numpy vers des natifs Python pour json.dumps."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def choose_calibration_method(y_train) -> str:
    # Heuristique simple : isotonic si assez de positifs, sinon sigmoid
    positives = int(np.sum(y_train == 1))
    return "isotonic" if positives >= 1000 else "sigmoid"


def best_threshold_by_f1(y_val, p_val) -> float:
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    idx = int(np.argmax(f1))
    return float(thr[idx])


def main():
    ap = argparse.ArgumentParser(description="Train + tuning (CV) pour Credit Scoring (logreg / xgb)")
    ap.add_argument("--model", choices=["logreg", "xgb", "all"], default="logreg")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=40,
                    help="Utile si tu passes à RandomizedSearchCV plus tard. Ignoré par Halving ici.")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=RANDOM_STATE_DEFAULT)
    ap.add_argument("--calibrate", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True,
                    help="Calibration des probabilités (isotonic/sigmoid selon données)")
    ap.add_argument("--refit_metric", choices=["roc_auc", "pr_auc"], default="roc_auc",
                    help="Métrique à optimiser durant le tuning")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    # ==== Données
    df, target, num_cols, cat_cols = load_data()
    X = df[[c for c in df.columns if c != target]]
    y = df[target].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # === MLflow (minimal) ===
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-scoring"))

    # Ratio pour scale_pos_weight (XGB)
    pos_ratio = (y_tr == 1).mean()
    neg_ratio = 1 - pos_ratio
    spw = max(neg_ratio / max(pos_ratio, 1e-9), 1e-6)

    # run_id pour versionner les artefacts
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Choix des modèles à entraîner
    models = ["logreg", "xgb"] if args.model == "all" else [args.model]

    # HalvingRandomSearchCV : une seule métrique en string
    scoring_str = "average_precision" if args.refit_metric == "pr_auc" else "roc_auc"

    # min_resources dynamique (~10% des échantillons, min 50)
    n_samples_train = X_tr.shape[0]
    min_resources_int = max(int(n_samples_train * 0.1), 50)
    if min_resources_int > n_samples_train:
        min_resources_int = max(1, n_samples_train // 2)

    for m in models:
        with mlflow.start_run(run_name=m):
            # Params globaux utiles
            mlflow.log_param("model_type", m)
            mlflow.log_param("cv_folds", args.cv)
            mlflow.log_param("refit_metric", args.refit_metric)
            mlflow.log_param("scoring_used", scoring_str)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("n_jobs", args.n_jobs)
            mlflow.log_param("halving_min_resources", int(min_resources_int))

            print(f"\n=== Tuning {m} (CV={args.cv}, Halving, refit={scoring_str}) ===")

            # pipeline + espace hyperparam
            pipe, space = build_pipe_and_space(m, num_cols, cat_cols, args.random_state, args.n_jobs)
            if m == "xgb":
                space["clf__scale_pos_weight"] = [spw * f for f in (0.5, 1.0, 1.5, 2.0)]

            cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)

            # HalvingRandomSearchCV corrigé:
            search = HalvingRandomSearchCV(
                estimator=pipe,
                param_distributions=space,
                factor=3,
                resource="n_samples",
                max_resources="auto",
                min_resources=min_resources_int,
                random_state=args.random_state,
                scoring=scoring_str,
                n_jobs=args.n_jobs,
                cv=cv,
                verbose=1,
            )

            search.fit(X_tr, y_tr)

            best_pipe: Pipeline = search.best_estimator_
            best_params = to_json_safe(search.best_params_)
            print(f"Best params ({m}): {best_params}")
            print(f"Best CV {scoring_str.upper()}: {search.best_score_:.4f}")

            # log des meilleurs hyperparamètres
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            # ==== Calibration ====
            final_pipe = best_pipe
            calib_method = None
            if args.calibrate:
                calib_method = choose_calibration_method(y_tr)
                mlflow.log_param("calibration_method", calib_method)
                print(f"Calibration des probabilités ({calib_method}, cv=3)...")
                final_pipe = CalibratedClassifierCV(estimator=best_pipe, method=calib_method, cv=3)
                final_pipe.fit(X_tr, y_tr)
            else:
                final_pipe.fit(X_tr, y_tr)

            # ==== Sélection de seuil sur validation interne (20% de X_tr) ====
            X_tune, X_thr, y_tune, y_thr = train_test_split(
                X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=args.random_state
            )

            thr_pipe = clone(best_pipe)
            if args.calibrate:
                thr_pipe = CalibratedClassifierCV(estimator=thr_pipe, method=calib_method, cv=3)
            thr_pipe.fit(X_tune, y_tune)

            proba_thr = thr_pipe.predict_proba(X_thr)[:, 1]
            thr = best_threshold_by_f1(y_thr, proba_thr)

            # ==== Évaluation hold-out ====
            proba_te = final_pipe.predict_proba(X_te)[:, 1]
            metrics = eval_metrics(y_te, proba_te, thr)

            # log métriques test + seuil
            mlflow.log_metric("test_auc",       metrics["auc"])
            mlflow.log_metric("test_pr_auc",    metrics["pr_auc"])
            mlflow.log_metric("test_f1",        metrics["f1"])
            mlflow.log_metric("test_precision", metrics["precision"])
            mlflow.log_metric("test_recall",    metrics["recall"])
            mlflow.log_metric("threshold_suggested", metrics["threshold"])

            print(
                f"Test ROC-AUC={metrics['auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
                f"F1={metrics['f1']:.3f}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
                f"(thr={metrics['threshold']:.3f})"
            )

            # --- Log du modèle au format MLflow (signature + exemple)
            try:
                input_example = X.head(5)
                signature = infer_signature(input_example, final_pipe.predict_proba(input_example)[:, 1])
                mlflow.sklearn.log_model(
                    sk_model=final_pipe,
                    artifact_path=f"model_{'logreg' if m=='logreg' else 'xgb'}",
                    signature=signature,
                    input_example=input_example,
                    # Pour activer le Model Registry plus tard, décommente :
                    # registered_model_name=os.getenv("MLFLOW_REGISTER_NAME", f"credit_scoring_{'logreg' if m=='logreg' else 'xgb'}")
                )
            except Exception as e:
                print(f"[WARN] MLflow log_model a échoué: {e!r}")

            # ==== Sauvegardes locales ====
            suffix = "logreg" if m == "logreg" else "xgb"
            model_fname = f"{run_id}_model_{suffix}.joblib"
            model_path = ART / model_fname
            dump(final_pipe, model_path)

            # Export cv_results_ pour audit
            cv_df = pd.DataFrame(search.cv_results_)
            cv_csv_path = ART / f"{run_id}_cv_results_{suffix}.csv"
            cv_df.to_csv(cv_csv_path, index=False)

            meta = {
                "model": (
                    "LogisticRegression" if m == "logreg" else "XGBoost"
                ) + (" (calibrated)" if args.calibrate else ""),
                "version": "1.4.2",
                "run_id": run_id,
                "features": list(X.columns),
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "metrics": metrics,
                "threshold_suggested": thr,
                "model_type": suffix,
                "cv": args.cv,
                "search": "HalvingRandomSearchCV",
                "refit_metric": args.refit_metric,
                "scoring_used": scoring_str,
                "calibrated": bool(args.calibrate),
                "calibration_method": calib_method,
                "best_params": best_params,
                "best_index": int(search.best_index_),
                "cv_mean_score": float(search.best_score_),
                "cv_csv": str(cv_csv_path),
                "halving_min_resources": int(min_resources_int),
                "mlflow": {
                    "tracking_uri": mlflow.get_tracking_uri(),
                    "experiment": os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-scoring"),
                    "run_id": mlflow.active_run().info.run_id
                },
                "lexicon": {
                    "duration":"Durée du crédit (mois)","credit_amount":"Montant du crédit","age":"Âge",
                    "credit_history":"Historique de crédit","checking_status":"Statut compte courant","savings_status":"Épargne",
                    "employment":"Ancienneté emploi","housing":"Logement","job":"Emploi (cat.)","purpose":"Objet du crédit",
                    "foreign_worker":"Travailleur étranger","installment_commitment":"Taux de mensualité / revenu",
                    "existing_credits":"Nb de crédits existants","residence_since":"Ancienneté de résidence",
                    "personal_status":"Statut personnel","other_parties":"Autres garants","other_payment_plans":"Autres plans",
                    "property_magnitude":"Patrimoine","num_dependents":"Personnes à charge","own_telephone":"Téléphone"
                }
            }
            meta_path = ART / f"{run_id}_metadata_{suffix}.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            # Log des artefacts dans MLflow (facilite l'audit)
            try:
                mlflow.log_artifact(str(cv_csv_path), artifact_path="artifacts")
                mlflow.log_artifact(str(meta_path),    artifact_path="artifacts")
            except Exception:
                pass

            # Explainer pour LR (non calibrée pour garder coef lisibles)
            if m == "logreg":
                expl_pipe = Pipeline([
                    ("pre", build_preproc(num_cols, cat_cols)),
                    ("clf", LogisticRegression(
                        max_iter=5000, class_weight="balanced", solver="saga",
                        C=best_params.get("clf__C", 1.0),
                        penalty=best_params.get("clf__penalty", "l2"),
                        l1_ratio=best_params.get("clf__l1_ratio", None),
                        n_jobs=args.n_jobs,
                        random_state=RANDOM_STATE_DEFAULT
                    ))
                ])
                # Fit global pour un explainer interprétable
                expl_pipe.fit(pd.concat([X_tr, X_te]), np.concatenate([y_tr, y_te]))
                dump(expl_pipe, ART / f"{run_id}_explainer_logreg.joblib")

            print(f"✔︎ Sauvé : {model_path.name} + {meta_path.name} + {cv_csv_path.name}")

    # Fin de la boucle
    print("\nTraining done ✅  (artifacts in artifacts/)")
    print("Disponibles :", ", ".join(sorted(p.name for p in ART.iterdir())))


if __name__ == "__main__":
    main()
