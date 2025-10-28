# src/utils/explain.py
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def _get_transformed_feature_names(preprocessor) -> List[str]:
    return preprocessor.get_feature_names_out().tolist()

def _parent_raw_feature(raw_token: str, original_cols: List[str]) -> str:
    """
    raw_token ressemble à:
      - 'age' (numérique) -> parent 'age'
      - 'purpose_radio/tv' (catégorielle encodée) -> parent 'purpose'
      - 'credit_history_critical/other existing credit' -> parent 'credit_history'
    On choisit le parent comme la PLUS LONGUE colonne d'origine qui est un préfixe de raw_token + '_' ou qui est exactement raw_token.
    """
    # tri par longueur décroissante pour éviter que 'credit' ne capture 'credit_history'
    for col in sorted(original_cols, key=len, reverse=True):
        if raw_token == col or raw_token.startswith(col + "_"):
            return col
    return raw_token  # fallback

def _aggregate_contributions_to_raw_features(
    transformed_names: List[str], contribs: np.ndarray, original_cols: List[str]
) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for name, val in zip(transformed_names, contribs):
        # name format: 'num__age' ou 'cat__credit_history_critical/...'
        parts = name.split("__", 1)
        raw = parts[1] if len(parts) == 2 else name
        parent = _parent_raw_feature(raw, original_cols)
        agg[parent] = agg.get(parent, 0.0) + float(val)
    return agg

def top3_factors_from_logreg_pipeline(pipe: Pipeline, x_df: pd.DataFrame, lexicon: Dict[str, str]) -> List[Dict]:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # valeurs transformées + contributions locales ~ coef * valeur_transfo
    Xtr = pre.transform(x_df)                     # (1, n_transformed)
    coefs = clf.coef_.ravel()                     # (n_transformed,)
    Xtr_arr = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
    contrib = (Xtr_arr * coefs).ravel()

    names = _get_transformed_feature_names(pre)
    original_cols = list(x_df.columns)

    agg = _aggregate_contributions_to_raw_features(names, contrib, original_cols)

    # top-3 par valeur absolue
    top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    res = []
    for raw, val in top:
        label = lexicon.get(raw, raw)
        direction = "augmente le risque" if val > 0 else "réduit le risque"
        res.append({"feature": label, "effect": direction})
    return res
