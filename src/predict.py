"""
predict.py
Generowanie kodów celnych dla rekordów z pustym STAWN (dane produkcyjne).
"""

import os
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Wczytanie konfiguracji ────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Wczytanie słowników ───────────────────────────────────────────────────────
def load_lookups() -> dict[str, list[str]]:
    lookup_dir = BASE_DIR / "data" / "lookups"
    lookups = {}
    for name in ("materialy", "pokrycia", "wglebienia"):
        path = lookup_dir / f"{name}.txt"
        lookups[name] = [
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return lookups


# ── Budowanie wiadomości ──────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are an expert in CN customs code classification. "
    "Based on the subgroup and product description, assign the correct 8-digit customs code. "
    "Return ONLY the 8-digit customs code. No explanation, no text, exactly 8 digits. "
    "Example: 73181499"
)


def build_messages(row: pd.Series, lookups: dict[str, list[str]]) -> list:
    user_content = (
        f"SUBGROUP: {row['PODGRUPA']}\n"
        f"MATNR: {row['MATNR']}\n"
        f"DESCRIPTION: {row['NAZWAPL']}"
    )
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content},
    ]


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
    model.eval()

    return model, tokenizer


# ── Predykcja z confidence ────────────────────────────────────────────────────
def predict_with_confidence(
    messages: list,
    model,
    tokenizer,
    max_new_tokens: int = 16,
) -> tuple[str, float]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Dekodowanie odpowiedzi
    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Confidence: średnie prawdopodobieństwo po tokenach kodu
    token_probs = [
        torch.softmax(score, dim=-1).max().item() for score in outputs.scores
    ]
    confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

    return decoded, round(confidence, 4)


# ── Walidacja formatu kodu ────────────────────────────────────────────────────
def is_valid_code(code: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", code))


# ── Główna logika ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    hub_cfg = cfg["hub"]

    # 1. Wczytanie danych produkcyjnych
    df = pd.read_csv(DATA_DIR / "model1_predict.csv", sep=";", dtype={"PODGRUPA": str})
    df = df[["MATNR", "OPIS", "PODGRUPA"]].dropna(subset=["OPIS"])
    print(f"Rekordy do predykcji: {len(df)}")

    # 2. Słowniki
    lookups = load_lookups()

    # 3. Model
    model, tokenizer = load_model_and_tokenizer(model_cfg, hub_cfg)

    # 4. Predykcje
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predykcja"):
        messages = build_messages(row, lookups)
        predicted, conf = predict_with_confidence(messages, model, tokenizer)

        records.append(
            {
                "MATNR": row["MATNR"],
                "PODGRUPA": row["PODGRUPA"],
                "OPIS": row["OPIS"],
                "STAWN_predykcja": predicted,
                "confidence": conf,
                "poprawny_format": is_valid_code(predicted),
            }
        )

    # 5. Zapis lokalny
    result_df = pd.DataFrame(records)
    result_df.to_csv(RESULTS_DIR / "predict_results.csv", index=False, sep=";")

    # 6. Upload na HF
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
