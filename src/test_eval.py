"""
test_eval.py
Szybki test modelu na 5 losowych próbkach z test setu.
Uruchom przed pełnym evaluate.py żeby sprawdzić czy model działa poprawnie.
"""

import os
import random
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"

N_SAMPLES = 5

SYSTEM_MSG = (
    "You are an expert in CN customs code classification. "
    "Based on the subgroup and product description, assign the correct 8-digit customs code. "
    "Return ONLY the 8-digit customs code. No explanation, no text, exactly 8 digits. "
    "Example: 73181499"
)


# ── Budowanie wiadomości ──────────────────────────────────────────────────────
def build_messages_for_predict(messages: list) -> list:
    """Zwraca messages bez ostatniego (assistant) dla inferencji."""
    return messages[:-1]


# ── Wczytanie konfiguracji ────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Ładowanie modelu ──────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_cfg: dict, hub_cfg: dict) -> tuple:
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

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        hub_cfg["hub_model_id"],
        token=os.getenv("HF_TOKEN"),
    )
    model = model.merge_and_unload()  # scala wagi LoRA z modelem bazowym
    model.eval()

    return model, tokenizer


# ── Predykcja ─────────────────────────────────────────────────────────────────
def predict(messages: list, model, tokenizer) -> str:
    text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


# ── Główna logika ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    hub_cfg = cfg["hub"]

    # 1. Wczytanie test setu i losowanie próbek
    dataset = load_dataset(
        data_cfg["hf_dataset"],
        split="test",
        token=os.getenv("HF_TOKEN"),
    )
    samples = random.sample(range(len(dataset)), N_SAMPLES)

    # 2. Model
    print("Ładowanie modelu...")
    model, tokenizer = load_model_and_tokenizer(model_cfg, hub_cfg)

    # 3. Test
    print(f"\n{'=' * 60}")
    print(f"QUICK TEST — {N_SAMPLES} losowych próbek z test setu")
    print(f"{'=' * 60}\n")

    correct = 0
    for i, idx in enumerate(samples):
        example = dataset[idx]
        messages = example["messages"]
        true_code = messages[-1]["content"]
        predicted = predict(messages, model, tokenizer)
        is_correct = predicted == true_code

        user_content = messages[1]["content"]
        matnr = user_content.split("\n")[1].replace("MATNR: ", "").strip()
        subgroup = user_content.split("\n")[0].replace("SUBGROUP: ", "").strip()

        status = "✓" if is_correct else "✗"
        correct += int(is_correct)

        print(f"[{i + 1}] {status} MATNR: {matnr} | PODGRUPA: {subgroup}")
        print(f"     Oczekiwany:  {true_code}")
        print(f"     Predykcja:   {predicted}\n")

    print(f"{'=' * 60}")
    print(f"Wynik: {correct}/{N_SAMPLES} poprawnych")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
