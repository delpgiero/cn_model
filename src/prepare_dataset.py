"""
prepare_dataset.py
Przygotowanie danych treningowych dla Modelu 1 (fine-tuning klasyfikacji kodów celnych).
"""

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LOOKUP_DIR = DATA_DIR / "lookups"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "baza_model1.csv"

# ── Stałe ────────────────────────────────────────────────────────────────────
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "twoj-username/customs-model1")
RANDOM_SEED = 42
VAL_SIZE = 0.1
TEST_SIZE = 0.1

SYSTEM_MSG = (
    "You are an expert in CN customs code classification. "
    "Based on the subgroup and product description, assign the correct 8-digit customs code. "
    "Return ONLY the 8-digit customs code. No explanation, no text, exactly 8 digits. "
    "Example: 73181499"
)


# ── Wczytanie słowników ───────────────────────────────────────────────────────
def load_lookups() -> dict[str, list[str]]:
    lookups = {}
    for name in ("materialy", "pokrycia", "wglebienia"):
        path = LOOKUP_DIR / f"{name}.txt"
        lookups[name] = [
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return lookups


# ── Budowanie wiadomości ──────────────────────────────────────────────────────
def build_messages(row: pd.Series, lookups: dict[str, list[str]]) -> dict:
    user_content = (
        f"SUBGROUP: {row['PODGRUPA']}\n"
        f"MATNR: {row['MATNR']}\n"
        f"DESCRIPTION: {row['NAZWAPL']}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(int(row["STAWN"]))},
        ]
    }


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
    df = pd.read_csv(CSV_PATH, sep=";", dtype={"PODGRUPA": str})

    # 2. Separacja: dane produkcyjne (puste STAWN → predykcja po treningu)
    df_predict = df[df["STAWN"].isna()].copy()
    df_predict.to_csv(OUTPUT_DIR / "model1_predict.csv", index=False, sep=";")
    print(f"Dane do predykcji: {len(df_predict)} rekordów")

    # 3. Dane dla Modelu 1
    df = df[df["STAWN"].notna()].copy()
    df["STAWN"] = df["STAWN"].astype(int).astype(str)
    df = df[["MATNR", "NAZWAPL", "PODGRUPA", "STAWN"]].dropna(subset=["NAZWAPL"])
    print(
        f"Model 1: {len(df)} rekordów | {df['STAWN'].nunique()} kodów | {df['PODGRUPA'].nunique()} podgrup"
    )

    # 4. Słowniki
    lookups = load_lookups()

    # 5. Split
    train, val, test = stratified_split(df)
    print(f"Split → train: {len(train)} | val: {len(val)} | test: {len(test)}")

    # 6. Budowanie messages i zapis lokalny
    for split_name, split_df in (("train", train), ("val", val), ("test", test)):
        split_df["messages"] = split_df.apply(
            lambda row: build_messages(row, lookups), axis=1
        )
        split_df.to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False, sep=";")

    # 7. Upload do HF
    def to_hf_dataset(split_df: pd.DataFrame) -> Dataset:
        records = split_df.apply(
            lambda row: build_messages(row, lookups), axis=1
        ).tolist()
        return Dataset.from_list(records)

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
