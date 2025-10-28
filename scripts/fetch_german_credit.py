import openml
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "german_credit.csv"

# Chargement du dataset German Credit (id=31)
d = openml.datasets.get_dataset(31)
df, *_ = d.get_data(dataset_format="dataframe")

# Identifier la colonne cible réelle
target_col = d.default_target_attribute
if target_col is None:
    # Tentative automatique si OpenML ne la fournit pas
    for c in df.columns:
        if c.lower() in ["class", "target", "credit_risk"]:
            target_col = c
            break

if target_col is None:
    raise ValueError(f"Aucune colonne cible trouvée dans le dataset ({df.columns.tolist()[:10]}...)")

print(f"[INFO] Colonne cible détectée : {target_col}")

# Normaliser le nom et encoder la cible
df = df.rename(columns={target_col: "target"})
df["target"] = (df["target"].astype(str).str.lower().str.strip() == "bad").astype(int)

df.to_csv(OUT, index=False)
print(f"✅ Données sauvegardées dans : {OUT.resolve()}")
print(df.head())

