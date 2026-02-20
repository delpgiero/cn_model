import os
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    model.eval()

    return model, tokenizer


# ── Predykcja jednego rekordu ─────────────────────────────────────────────────
def predict(messages: list, model, tokenizer, max_new_tokens: int = 16) -> str:
    text = tokenizer.apply_chat_template(
        messages[:-1],  # system + user, bez assistant
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
        )

    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    return decoded


# ── Walidacja formatu kodu ────────────────────────────────────────────────────
def is_valid_code(code: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", code))


# ── Główna logika ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    hub_cfg = cfg["hub"]

    # 1. Wczytanie test setu
    dataset = load_dataset(
        data_cfg["hf_dataset"],
        split="test",
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Test set: {len(dataset)} rekordów")

    # 2. Model
    model, tokenizer = load_model_and_tokenizer(model_cfg, hub_cfg)

    # 3. Predykcje
    records = []
    for example in tqdm(dataset, desc="Ewaluacja"):
        messages = example["messages"]
        true_code = messages[-1]["content"]
        predicted = predict(messages, model, tokenizer)
        valid_format = is_valid_code(predicted)

        user_content = messages[1]["content"]
        subgroup = user_content.split("\n")[0].replace("SUBGROUP: ", "").strip()
        matnr = user_content.split("\n")[1].replace("MATNR: ", "").strip()

        records.append(
            {
                "MATNR": matnr,
                "PODGRUPA": subgroup,
                "STAWN_prawdziwy": true_code,
                "STAWN_predykcja": predicted,
                "poprawny": predicted == true_code,
                "poprawny_format": valid_format,
            }
        )

    # 4. Wyniki globalne
    df = pd.DataFrame(records)
    accuracy_global = df["poprawny"].mean()
    format_accuracy = df["poprawny_format"].mean()
    print(f"\nAccuracy globalna:  {accuracy_global:.4f} ({accuracy_global * 100:.2f}%)")
    print(f"Poprawny format:    {format_accuracy:.4f} ({format_accuracy * 100:.2f}%)")

    # 5. Accuracy per podgrupa
    per_group = (
        df.groupby("PODGRUPA")["poprawny"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy")
    )
    print(f"\nNajsłabsze podgrupy:\n{per_group.head(10).to_string()}")

    # 6. Zapis wyników lokalnie
    df.to_csv(RESULTS_DIR / "eval_results.csv", index=False, sep=";")
    per_group.to_csv(RESULTS_DIR / "eval_per_group.csv", sep=";")

    # 7. Upload wyników na HF
    from huggingface_hub import HfApi

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
