"""
train.py
Fine-tuning Qwen2.5-3B-Instruct na danych klasyfikacji kodów celnych (Model 1).
"""

import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"


# ── Wczytanie konfiguracji ────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Konfiguracja kwantyzacji 4-bit ────────────────────────────────────────────
def get_bnb_config(qlora_cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ── Ładowanie modelu i tokenizera ─────────────────────────────────────────────
def load_model_and_tokenizer(
    model_cfg: dict,
    bnb_config: BitsAndBytesConfig,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    return model, tokenizer


# ── Konfiguracja LoRA ─────────────────────────────────────────────────────────
def get_lora_config(qlora_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["lora_alpha"],
        lora_dropout=qlora_cfg["lora_dropout"],
        target_modules=qlora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


# ── Formatowanie datasetu ─────────────────────────────────────────────────────
def format_messages(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Zamienia listę messages na pojedynczy string używając chat template Qwen.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ── Główna logika ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    qlora_cfg = cfg["qlora"]
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    hub_cfg = cfg["hub"]

    # 1. Dataset
    dataset = load_dataset(
        data_cfg["hf_dataset"],
        token=os.getenv("HF_TOKEN"),
    )

    # 2. Model i tokenizer
    bnb_config = get_bnb_config(qlora_cfg)
    model, tokenizer = load_model_and_tokenizer(model_cfg, bnb_config)

    # 3. Formatowanie datasetu
    dataset = dataset.map(
        lambda ex: format_messages(ex, tokenizer),
        remove_columns=["messages"],
    )

    # 4. LoRA
    lora_config = get_lora_config(qlora_cfg)

    # 5. Training arguments + SFT config
    training_args = SFTConfig(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=training_cfg["warmup_steps"],
        weight_decay=training_cfg["weight_decay"],
        fp16=training_cfg["fp16"],
        bf16=training_cfg["bf16"],
        logging_steps=training_cfg["logging_steps"],
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        eval_strategy="steps",
        report_to="tensorboard",
        hub_token=os.getenv("HF_TOKEN"),
        push_to_hub=hub_cfg["push_to_hub"],
        hub_model_id=hub_cfg["hub_model_id"],
        max_length=model_cfg["max_seq_length"],
    )

    # 6. Trener
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # 7. Trening
    trainer.train()

    # 8. Zapis finalnego modelu
    trainer.save_model(training_cfg["output_dir"])
    if hub_cfg["push_to_hub"]:
        trainer.push_to_hub()
        print(f"Model wgrany: {hub_cfg['hub_model_id']}")


if __name__ == "__main__":
    main()
