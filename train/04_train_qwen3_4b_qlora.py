#!/usr/bin/env python
"""
Train Qwen3-4B-Instruct-2507 with QLoRA on labeled email summaries (5-line schema).
Optimized for 4GB VRAM (GTX 1650 Ti).
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train QLoRA student model (local)")
    parser.add_argument("--data_path", default="data/labeled/train.jsonl")
    parser.add_argument("--output_dir", default="train/output")
    parser.add_argument("--model_name", default="models/qwen3-4b-instruct-2507")
    parser.add_argument("--resume_from", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--max_seq_length", type=int, default=1536)  # safer for 4GB
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    model_path = Path(args.model_name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}\n"
            f"Expected local model at: models/qwen3-4b-instruct-2507"
        )

    # 4-bit QLoRA config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    use_fast=False,
   )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA settings tuned for 4GB VRAM (stable)
    lora_config = LoraConfig(
        r=8,                      # smaller rank to fit low VRAM
        lora_alpha=16,            # scale with r
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Load labeled chat-format dataset
    dataset = load_dataset("json", data_files=str(data_path))["train"]

    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    # Remove original columns to reduce memory usage
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",       # transformers>=5 uses eval_strategy
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        group_by_length=True,     # speeds up + helps stability
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Training complete. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
