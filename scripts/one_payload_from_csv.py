# scripts/one_payload_from_csv.py
import json, sys
import pandas as pd

df = pd.read_csv("data/raw/german_credit.csv")
row = df.iloc[0].to_dict()  # prends la première ligne
target = row.pop("target", None)  # on retire la cible
print(json.dumps({"payload": row}, ensure_ascii=False, indent=2))
