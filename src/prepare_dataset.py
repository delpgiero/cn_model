"""
prepare_dataset.py
Przygotowanie danych treningowych dla Modelu 1 - podejście klasyfikacyjne.
Input: PODGRUPA + MATNR + AVAILABLE (dostępne labele w podgrupie)
Output: label (int) mapowany na kod celny
"""

import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "baza_model1.csv"
LABEL_MAP_PATH = OUTPUT_DIR / "label_map.json"

# ── Stałe ────────────────────────────────────────────────────────────────────
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "twoj-username/customs-model1")
RANDOM_SEED = 42
VAL_SIZE = 0.1
TEST_SIZE = 0.1


# ── Budowanie tekstu wejściowego ──────────────────────────────────────────────
def build_text(row: pd.Series, subgroup_labels: dict) -> str:
    """Format: PODGRUPA: 1102 | MATNR: 1102100130S10D931T | NAZWAPL: Śruba M10 | AVAILABLE: 0,5,12"""
    available = ",".join(str(l) for l in subgroup_labels[row["PODGRUPA"]])
    return f"PODGRUPA: {row['PODGRUPA']} | MATNR: {row['MATNR']} | NAZWAPL: {row['NAZWAPL']} | AVAILABLE: {available}"


# ── Split stratyfikowany ──────────────────────────────────────────────────────
def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat_col = df["PODGRUPA"].astype(str) + "_" + df["STAWN"].astype(str)
    min_count = strat_col.value_counts()
    valid_mask = strat_col.isin(min_count[min_count >= 2].index)

    df_valid = df[valid_mask]
    df_singles = df[~valid_mask]

    train_val, test = train_test_split(
        df_valid,
        test_size=TEST_SIZE,
        stratify=strat_col[valid_mask],
        random_state=RANDOM_SEED,
    )

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    strat_col_tv = (
        train_val["PODGRUPA"].astype(str) + "_" + train_val["STAWN"].astype(str)
    )

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=strat_col_tv,
        random_state=RANDOM_SEED,
    )

    train = pd.concat([train, df_singles], ignore_index=True)
    return train, val, test


# ── Główna logika ─────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Wczytanie danych
    df = pd.read_csv(CSV_PATH, sep=";", dtype={"PODGRUPA": str, "STAWN": str})

    # 2. Dane produkcyjne (puste STAWN)
    df_predict = df[df["STAWN"].isna()].copy()
    df_predict.to_csv(OUTPUT_DIR / "model1_predict.csv", index=False, sep=";")
    print(f"Dane do predykcji: {len(df_predict)} rekordów")

    # 3. Dane treningowe
    df = df[df["STAWN"].notna()].copy()
    df["STAWN"] = df["STAWN"].astype(float).astype(int).astype(str)
    df = df[["MATNR", "PODGRUPA", "NAZWAPL", "STAWN"]].dropna()
    print(
        f"Model 1: {len(df)} rekordów | {df['STAWN'].nunique()} kodów | {df['PODGRUPA'].nunique()} podgrup"
    )

    # 4. Label mapping: kod celny → int (globalny, posortowany)
    unique_codes = sorted(df["STAWN"].unique())
    code_to_label = {code: idx for idx, code in enumerate(unique_codes)}
    label_to_code = {str(idx): code for code, idx in code_to_label.items()}

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"code_to_label": code_to_label, "label_to_code": label_to_code},
            f,
            indent=2,
        )
    print(f"Label map: {len(unique_codes)} unikalnych kodów → {LABEL_MAP_PATH}")

    # 5. Mapowanie per podgrupa (dostępne labele w danej podgrupie)
    subgroup_labels = (
        df.groupby("PODGRUPA")["STAWN"]
        .apply(lambda x: [code_to_label[c] for c in sorted(x.unique())])
        .to_dict()
    )
    with open(OUTPUT_DIR / "subgroup_labels.json", "w", encoding="utf-8") as f:
        json.dump(subgroup_labels, f, indent=2)
    print(f"Subgroup labels zapisane: {len(subgroup_labels)} podgrup")

    # 6. Dodanie labeli i tekstu (z dostępnymi labelami w prompcie)
    df["label"] = df["STAWN"].map(code_to_label)
    df["text"] = df.apply(lambda row: build_text(row, subgroup_labels), axis=1)

    # 7. Split
    train, val, test = stratified_split(df)
    print(f"Split → train: {len(train)} | val: {len(val)} | test: {len(test)}")

    # 8. Zapis lokalny CSV
    for split_name, split_df in (("train", train), ("val", val), ("test", test)):
        split_df[["text", "label", "STAWN"]].to_csv(
            OUTPUT_DIR / f"{split_name}.csv", index=False, sep=";"
        )
    print(f"CSV zapisane w: {OUTPUT_DIR}")

    # 9. Upload do HF
    def to_hf_dataset(split_df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "text": split_df["text"].tolist(),
                "label": split_df["label"].tolist(),
            }
        )

    dataset = DatasetDict(
        {
            "train": to_hf_dataset(train),
            "validation": to_hf_dataset(val),
            "test": to_hf_dataset(test),
        }
    )

    dataset.push_to_hub(
        HF_DATASET_ID,
        token=os.getenv("HF_TOKEN"),
        private=True,
    )
    print(f"Dataset wgrany: {HF_DATASET_ID}")


if __name__ == "__main__":
    main()
