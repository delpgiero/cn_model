"""
test_eval.py
Szybki test klasyfikatora na 5 losowych próbkach z test setu.
Stosuje maskowanie logitów per podgrupa.
"""

import json
import os
import random
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"
LABEL_MAP_PATH = BASE_DIR / "data" / "processed" / "label_map.json"
SUBGROUP_PATH = BASE_DIR / "data" / "processed" / "subgroup_labels.json"

N_SAMPLES = 5


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


# ── Predykcja z maskowaniem per podgrupa ─────────────────────────────────────
def predict(
    text: str,
    podgrupa: str,
    model,
    tokenizer,
    subgroup_labels: dict,
    label_to_code: dict,
) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]  # shape: (num_labels,)

    # Maskowanie: zostaw tylko kody dostępne w podgrupie
    allowed = subgroup_labels.get(podgrupa, list(range(logits.shape[0])))
    mask = torch.full_like(logits, float("-inf"))
    mask[allowed] = logits[allowed]

    predicted_label = mask.argmax().item()
    return label_to_code[str(predicted_label)]


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

    with open(SUBGROUP_PATH, encoding="utf-8") as f:
        subgroup_labels = json.load(f)

    num_labels = len(label_to_code)

    # 2. Dataset
    dataset = load_dataset(
        data_cfg["hf_dataset"],
        split="test",
        token=os.getenv("HF_TOKEN"),
    )
    samples = random.sample(range(len(dataset)), N_SAMPLES)

    # 3. Model
    print("Ładowanie modelu...")
    model, tokenizer = load_model_and_tokenizer(model_cfg, hub_cfg, num_labels)

    # 4. Test
    print(f"\n{'=' * 60}")
    print(f"QUICK TEST — {N_SAMPLES} losowych próbek z test setu")
    print(f"{'=' * 60}\n")

    correct = 0
    for i, idx in enumerate(samples):
        example = dataset[idx]
        text = example["text"]
        true_label = example["label"]
        true_code = label_to_code[str(true_label)]

        # Wyciągnij podgrupę z tekstu
        podgrupa = text.split(" | ")[0].replace("PODGRUPA: ", "").strip()
        predicted = predict(
            text, podgrupa, model, tokenizer, subgroup_labels, label_to_code
        )

        is_correct = predicted == true_code
        correct += int(is_correct)
        status = "✓" if is_correct else "✗"

        print(f"[{i + 1}] {status} {text}")
        print(f"     Oczekiwany:  {true_code}")
        print(f"     Predykcja:   {predicted}\n")

    print(f"{'=' * 60}")
    print(f"Wynik: {correct}/{N_SAMPLES} poprawnych")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
