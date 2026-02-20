"""
predict.py
Generowanie kodów celnych dla rekordów z pustym STAWN (dane produkcyjne).
Podejście klasyfikacyjne z logit masking per podgrupa.
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"
LABEL_MAP_PATH = BASE_DIR / "data" / "processed" / "label_map.json"
SUBGROUP_PATH = BASE_DIR / "data" / "processed" / "subgroup_labels.json"
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICT_BATCH_SIZE = 128


# ── Konfiguracja ──────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Ładowanie modelu ──────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_cfg: dict, hub_cfg: dict, num_labels: int) -> tuple:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hub_cfg["hub_model_id"],
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(
        base_model,
        hub_cfg["hub_model_id"],
        token=os.getenv("HF_TOKEN"),
    )
    model.eval()

    return model, tokenizer


# ── Budowanie tekstu wejściowego ──────────────────────────────────────────────
def build_text(row: pd.Series, subgroup_labels: dict) -> str:
    podgrupa = str(row["PODGRUPA"])
    available = ",".join(str(l) for l in subgroup_labels.get(podgrupa, []))
    return (
        f"PODGRUPA: {podgrupa} | MATNR: {row['MATNR']} | "
        f"NAZWAPL: {row['NAZWAPL']} | AVAILABLE: {available}"
    )


# ── Maskowanie logitów per podgrupa ──────────────────────────────────────────
def apply_subgroup_mask(
    logits: torch.Tensor,
    podgrupy: list[str],
    subgroup_labels: dict,
) -> torch.Tensor:
    masked = torch.full_like(logits, float("-inf"))
    for i, podgrupa in enumerate(podgrupy):
        allowed = subgroup_labels.get(podgrupa, list(range(logits.shape[1])))
        masked[i, allowed] = logits[i, allowed]
    return masked


# ── Główna logika ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    hub_cfg = cfg["hub"]

    # 1. Label map + subgroup labels
    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map = json.load(f)
    label_to_code = label_map["label_to_code"]
    num_labels = len(label_to_code)

    with open(SUBGROUP_PATH, encoding="utf-8") as f:
        subgroup_labels = json.load(f)

    # 2. Dane produkcyjne
    df = pd.read_csv(
        DATA_DIR / "model1_predict.csv",
        sep=";",
        dtype={"PODGRUPA": str},
    )
    df = df[["MATNR", "NAZWAPL", "PODGRUPA"]].dropna(subset=["MATNR", "PODGRUPA"])
    print(f"Rekordy do predykcji: {len(df)}")

    # 3. Model
    print("Ładowanie modelu...")
    model, tokenizer = load_model_and_tokenizer(model_cfg, hub_cfg, num_labels)

    # 4. Budowanie tekstów
    df["text"] = df.apply(lambda row: build_text(row, subgroup_labels), axis=1)

    # 5. Tokenizacja
    dataset = Dataset.from_dict({"text": df["text"].tolist()})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["model"]["max_seq_length"],
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized, batch_size=PREDICT_BATCH_SIZE, collate_fn=collator
    )

    # 6. Batch inference
    all_predictions = []
    all_confidences = []
    all_texts = df["text"].tolist()

    print("Predykcja...")
    text_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batch inference"):
            batch_size = batch["input_ids"].shape[0]
            podgrupy = [
                all_texts[text_idx + i]
                .split(" | ")[0]
                .replace("PODGRUPA: ", "")
                .strip()
                for i in range(batch_size)
            ]
            text_idx += batch_size

            inputs = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**inputs).logits

            masked = apply_subgroup_mask(logits, podgrupy, subgroup_labels)
            probs = torch.softmax(masked, dim=-1)
            predictions = probs.argmax(dim=-1).cpu().tolist()
            confidences = probs.max(dim=-1).values.cpu().tolist()

            all_predictions.extend(predictions)
            all_confidences.extend(confidences)

    # 7. Wyniki
    df["STAWN_predykcja"] = [label_to_code[str(p)] for p in all_predictions]
    df["confidence"] = [round(c, 4) for c in all_confidences]
    df = df.drop(columns=["text"])

    result_df = df[["MATNR", "PODGRUPA", "NAZWAPL", "STAWN_predykcja", "confidence"]]
    result_df.to_csv(RESULTS_DIR / "predict_results.csv", index=False, sep=";")
    print(f"Wyniki zapisane: {RESULTS_DIR / 'predict_results.csv'}")

    # 8. Upload na HF
    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=str(RESULTS_DIR / "predict_results.csv"),
        path_in_repo="results/predict_results.csv",
        repo_id=hub_cfg["hub_model_id"],
        repo_type="model",
    )
    print(f"Wyniki wgrane na HF: {hub_cfg['hub_model_id']}")


if __name__ == "__main__":
    main()
