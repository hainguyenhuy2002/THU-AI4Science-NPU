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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, get_cosine_schedule_with_warmup


SYSTEM_PROMPT = "You are a careful clinical pharmacy assistant. Give concise and safe answers."


def require_npu() -> torch.device:
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available. Check CANN, torch-npu, and ASCEND_VISIBLE_DEVICES.")
    return torch.device("npu:0")


class JsonlSftDataset(Dataset):
    def __init__(self, path: str, tokenizer, cutoff_len: int, system_prompt: str):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.system_prompt = system_prompt
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError(f"No training examples in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        user = row["instruction"] if not row.get("input") else f"{row['instruction']}\n\n{row['input']}"
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = row["output"].strip()
        messages.append({"role": "assistant", "content": answer})
        full = self.tokenizer.apply_chat_template(messages, tokenize=False)
        if not answer or not full.startswith(prompt):
            raise ValueError(f"Cannot locate the assistant answer in example {index + 1}")
        encoded = self.tokenizer(
            full, add_special_tokens=False, truncation=True, max_length=self.cutoff_len, return_offsets_mapping=True,
        )
        labels = [
            token if end > len(prompt) else -100
            for token, (_, end) in zip(encoded.input_ids, encoded.offset_mapping)
        ]
        if all(label == -100 for label in labels):
            raise ValueError(f"The assistant answer in example {index + 1} was truncated; increase --cutoff-len")
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


def load_model(model_path: str, config, gguf_path: Path | None):
    options = {"dtype": torch.bfloat16, "attn_implementation": "eager", "trust_remote_code": True}
    if gguf_path is None:
        return AutoModelForCausalLM.from_pretrained(model_path, config=config, **options)

    from gguf import GGUFReader

    gguf_path = gguf_path.resolve(strict=True)
    gguf_arch = GGUFReader(gguf_path).get_field("general.architecture").contents()
    if config.model_type == "mistral" and gguf_arch == "llama":
        if config.sliding_window is not None:
            raise ValueError("This GGUF import requires a Mistral model without sliding-window attention")
        # Mistral Small 2501 uses the Llama tensor layout in GGUF. Retain the
        # upstream config values, including its explicit attention head size.
        config_values = config.to_dict()
        del config_values["model_type"]
        config_values.pop("architectures", None)
        config = LlamaConfig(**config_values)

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, gguf_file=str(gguf_path), output_loading_info=True, **options,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"GGUF weights do not match the supplied model configuration: {loading_info}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gguf-file", type=Path, help="Dequantize this local GGUF using --model for config and tokenizer")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--target-modules", nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Module names for LoRA, or all-linear to select supported linear layers automatically",
    )
    parser.add_argument("--data", default="data/medical_flashcards_lora.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-0.5b-medical-lora")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=200)
    args = parser.parse_args()
    if min(args.max_steps, args.grad_accum, args.cutoff_len, args.batch_size, args.save_steps) < 1:
        parser.error("Steps, gradient accumulation, cutoff length, and batch size must be positive")

    device = require_npu()
    print(f"Using device: {device}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer_options = {"fix_mistral_regex": True} if config.model_type == "mistral" else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, **tokenizer_options)
    if not tokenizer.is_fast:
        raise ValueError("This trainer requires a fast tokenizer to locate assistant tokens")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = JsonlSftDataset(args.data, tokenizer, args.cutoff_len, args.system_prompt)
    # Check the chat template and cutoff before loading the model weights.
    dataset[0]
    model = load_model(args.model, config, args.gguf_file)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear" if args.target_modules == ["all-linear"] else args.target_modules,
        ),
    )
    model.to(device)
    try:
        print(f"NPU memory allocated after model.to: {torch.npu.memory_allocated() / 1024 / 1024:.1f} MiB")
    except Exception as exc:
        print(f"NPU memory allocated after model.to: unavailable ({exc})")
    model.train()
    model.print_trainable_parameters()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    updates = math.ceil(args.max_steps / args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * 0.03), updates)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "training_metrics.jsonl"
    step = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=args.max_steps, desc="LoRA fine-tuning")
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            group_start = step - step % args.grad_accum
            group_size = min(args.grad_accum, args.max_steps - group_start)
            (loss / group_size).backward()
            step += 1
            if step % args.grad_accum == 0 or step == args.max_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            progress.update(1)
            if step == 1 or step % 10 == 0 or step == args.max_steps:
                raw_loss = float(loss.detach().cpu())
                if not math.isfinite(raw_loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: {raw_loss}")
                with metrics.open("a", encoding="utf-8") as stream:
                    row = {"step": step, "optimizer_steps": optimizer_steps,
                           "loss": raw_loss, "device": str(device)}
                    stream.write(json.dumps(row) + "\n")
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
