"""
evaluate.py
Pełna ewaluacja klasyfikatora kodów celnych na test secie.
Batch inference dla maksymalnego wykorzystania GPU.
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
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
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_BATCH_SIZE = 128  # batch inference — większy niż trening


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
    data_cfg = cfg["data"]
    hub_cfg = cfg["hub"]

    # 1. Label map + subgroup labels
    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map = json.load(f)
    label_to_code = label_map["label_to_code"]
    num_labels = len(label_to_code)

    with open(SUBGROUP_PATH, encoding="utf-8") as f:
        subgroup_labels = json.load(f)

    # 2. Dataset
    dataset = load_dataset(
        data_cfg["hf_dataset"],
        split="test",
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Test set: {len(dataset)} rekordów")

    # 3. Model
    print("Ładowanie modelu...")
    model, tokenizer = load_model_and_tokenizer(model_cfg, hub_cfg, num_labels)

    # 4. Tokenizacja
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
    dataloader = DataLoader(tokenized, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

    # 5. Batch inference
    all_predictions = []
    all_labels = []
    all_texts = dataset["text"]  # potrzebne do wyciągnięcia podgrupy

    print("Ewaluacja...")
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

            inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits

            # Maskowanie per podgrupa
            masked = apply_subgroup_mask(logits, podgrupy, subgroup_labels)
            predictions = masked.argmax(dim=-1).cpu().tolist()

            all_predictions.extend(predictions)
            all_labels.extend(batch["labels"].tolist())

    # 6. Wyniki
    records = []
    for i, (pred_label, true_label) in enumerate(zip(all_predictions, all_labels)):
        text = all_texts[i]
        podgrupa = text.split(" | ")[0].replace("PODGRUPA: ", "").strip()
        matnr = text.split(" | ")[1].replace("MATNR: ", "").strip()
        true_code = label_to_code[str(true_label)]
        pred_code = label_to_code[str(pred_label)]

        records.append(
            {
                "MATNR": matnr,
                "PODGRUPA": podgrupa,
                "STAWN_prawdziwy": true_code,
                "STAWN_predykcja": pred_code,
                "poprawny": pred_code == true_code,
            }
        )

    df = pd.DataFrame(records)
    accuracy_global = df["poprawny"].mean()
    print(f"\nAccuracy globalna: {accuracy_global:.4f} ({accuracy_global * 100:.2f}%)")

    # 7. Accuracy per podgrupa
    per_group = (
        df.groupby("PODGRUPA")["poprawny"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy")
    )
    print(f"\nNajsłabsze podgrupy:\n{per_group.head(10).to_string()}")

    # 8. Zapis lokalny
    df.to_csv(RESULTS_DIR / "eval_results.csv", index=False, sep=";")
    per_group.to_csv(RESULTS_DIR / "eval_per_group.csv", sep=";")

    # 9. Upload na HF
    api = HfApi(token=os.getenv("HF_TOKEN"))
    for filename in ("eval_results.csv", "eval_per_group.csv"):
        api.upload_file(
            path_or_fileobj=str(RESULTS_DIR / filename),
            path_in_repo=f"results/{filename}",
            repo_id=hub_cfg["hub_model_id"],
            repo_type="model",
        )
    print(f"Wyniki wgrane na HF: {hub_cfg['hub_model_id']}")


if __name__ == "__main__":
    main()
