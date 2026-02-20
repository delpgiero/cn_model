"""
train.py
Fine-tuning Qwen2.5-3B jako klasyfikator kodów celnych (64 klasy).
Używa QLoRA + AutoModelForSequenceClassification.
"""

import os
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import evaluate

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"


# ── Konfiguracja ──────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Kwantyzacja 4-bit ─────────────────────────────────────────────────────────
def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ── Model + tokenizer ─────────────────────────────────────────────────────────
def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    bnb_config: BitsAndBytesConfig,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Wymagane przy 4-bit + gradient checkpointing
    model = prepare_model_for_kbit_training(model)
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ── LoRA ──────────────────────────────────────────────────────────────────────
def get_lora_config(qlora_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["lora_alpha"],
        lora_dropout=qlora_cfg["lora_dropout"],
        target_modules=qlora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.SEQ_CLS,  # sequence classification
    )


# ── Metryki ───────────────────────────────────────────────────────────────────
def get_compute_metrics():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    return compute_metrics


# ── Główna logika ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    qlora_cfg = cfg["qlora"]
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    hub_cfg = cfg["hub"]

    # 1. Dataset z HF
    dataset = load_dataset(
        data_cfg["hf_dataset"],
        token=os.getenv("HF_TOKEN"),
    )
    num_labels = (
        dataset["train"].features["label"].num_classes
        if hasattr(dataset["train"].features["label"], "num_classes")
        else len(set(dataset["train"]["label"]))
    )

    print(f"Liczba klas: {num_labels}")

    # 2. Model + tokenizer
    bnb_config = get_bnb_config()
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["name"], num_labels, bnb_config
    )

    # 3. LoRA
    lora_config = get_lora_config(qlora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Tokenizacja datasetu
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=model_cfg["max_seq_length"],
            padding=False,  # padding robi DataCollator
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=training_cfg["warmup_steps"],
        weight_decay=training_cfg["weight_decay"],
        bf16=training_cfg["bf16"],
        fp16=training_cfg["fp16"],
        logging_steps=training_cfg["logging_steps"],
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        greater_is_better=training_cfg["greater_is_better"],
        eval_strategy="steps",
        report_to="tensorboard",
        hub_token=os.getenv("HF_TOKEN"),
        push_to_hub=hub_cfg["push_to_hub"],
        hub_model_id=hub_cfg["hub_model_id"],
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(),
    )

    # 7. Trening
    trainer.train()

    # 8. Zapis + push
    trainer.save_model(training_cfg["output_dir"])
    if hub_cfg["push_to_hub"]:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
