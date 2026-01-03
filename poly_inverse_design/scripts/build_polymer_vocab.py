import sys
import os
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_ner_csvs():
    train = pd.read_csv(PROJECT_ROOT / "ner_train.csv")
    dev = pd.read_csv(PROJECT_ROOT / "ner_dev.csv")
    test = pd.read_csv(PROJECT_ROOT / "ner_test.csv")
    return pd.concat([train, dev, test], ignore_index=True)

def build_polymer_vocab(df):
    polymers = df[df["entity_group"] == "POLYMER"].copy()
    polymers["polymer_norm"] = polymers["entity_text"].str.strip().str.lower()

    vocab = (
        polymers.groupby("polymer_norm")
        .agg(
            frequency=("polymer_norm", "size"),
            example=("entity_text", "first"),
            n_variants=("entity_text", "nunique")
        )
        .reset_index()
        .rename(columns={"polymer_norm": "polymer_name"})
        .sort_values("frequency", ascending=False)
    )
    return vocab

if __name__ == "__main__":
    df = load_ner_csvs()
    vocab = build_polymer_vocab(df)

    out_path = PROCESSED_DIR / "polymer_vocab.csv"
    vocab.to_csv(out_path, index=False)

    print(f"Saved polymer vocabulary to {out_path}")
    print(f"Unique polymers: {len(vocab)}")
    print(vocab.head(20))
