#!/usr/bin/env python3
"""Minimal LoRA SFT trainer for Ascend NPU."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


SYSTEM_PROMPT = "You are a careful clinical pharmacy assistant. Give concise and safe answers."


def require_npu() -> torch.device:
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available. Check CANN, torch-npu, and ASCEND_VISIBLE_DEVICES.")
    return torch.device("npu:0")


class JsonlSftDataset(Dataset):
    def __init__(self, path: str, tokenizer, cutoff_len: int):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        user = row["instruction"] if not row.get("input") else f"{row['instruction']}\n\n{row['input']}"
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        full = f"{prompt}{row['output']}<|im_end|>"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        encoded = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.cutoff_len)
        labels = encoded.input_ids.copy()
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": encoded.input_ids, "attention_mask": encoded.attention_mask, "labels": labels}


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad = max_len - len(item["input_ids"])
        out["input_ids"].append(item["input_ids"] + [pad_token_id] * pad)
        out["attention_mask"].append(item["attention_mask"] + [0] * pad)
        out["labels"].append(item["labels"] + [-100] * pad)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/medical_flashcards_lora.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-0.5b-medical-lora")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=200)
    args = parser.parse_args()

    device = require_npu()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    dataset = JsonlSftDataset(args.data, tokenizer, args.cutoff_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, math.ceil(args.max_steps * 0.03)), args.max_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "training_metrics.jsonl"
    step = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=args.max_steps, desc="LoRA fine-tuning")
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            if step % 10 == 0:
                raw_loss = float(loss.detach().cpu()) * args.grad_accum
                metrics.open("a", encoding="utf-8").write(json.dumps({"step": step, "loss": raw_loss}) + "\n")
                progress.set_postfix(loss=f"{raw_loss:.4f}")
            if step % args.save_steps == 0:
                ckpt = out_dir / f"checkpoint-{step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
            if step >= args.max_steps:
                break
    progress.close()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
