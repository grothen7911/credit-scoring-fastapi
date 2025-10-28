import pandas as pd
from pathlib import Path

p = Path("logs/requests.jsonl")
if not p.exists():
    print("No log file.")
    raise SystemExit

df = pd.read_json(p, lines=True)
print(df.tail(3))
print("\n--- Metrics ---")
print("Count:", len(df))
print("Accept rate:", (df["decision"] == "ACCEPT").mean())
print("Avg probability:", df["probability"].mean())
print("\nBy day:")
if "timestamp_utc" in df:
    df["day"] = pd.to_datetime(df["timestamp_utc"]).dt.date
    print(df.groupby("day")["decision"].apply(lambda s: (s=="ACCEPT").mean()))
