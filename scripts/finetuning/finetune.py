#!/usr/bin/env python3
"""Minimal LoRA SFT trainer for Ascend NPU."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, get_cosine_schedule_with_warmup

from exp_setup.distributed import npu_run
from exp_setup.finetune_data import JsonlSftDataset, collate


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
    parser.add_argument("--system-prompt", default="Answer concisely.")
    parser.add_argument(
        "--target-modules", nargs="+",
        default=["all-linear"],
        help="Module names for LoRA, or all-linear to select supported linear layers automatically",
    )
    parser.add_argument("--data", default="exp_setup/finetuning.jsonl")
    parser.add_argument("--output-dir", default="outputs/finetuning")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--cutoff-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=200)
    args = parser.parse_args()
    if min(args.max_steps, args.grad_accum, args.cutoff_len, args.batch_size, args.save_steps) < 1:
        parser.error("Steps, gradient accumulation, cutoff length, and batch size must be positive")

    with npu_run() as (device, rank, world_size):
        finetune(args, device, rank, world_size)


def finetune(args, device, rank, world_size):
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
    adapter = model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(adapter, device_ids=[device.index], broadcast_buffers=False)
    model.train()
    if rank == 0:
        adapter.print_trainable_parameters()

    sampler = DistributedSampler(dataset) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    updates = math.ceil(args.max_steps / args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * 0.03), updates)

    out_dir = Path(args.output_dir)
    metrics = out_dir / "training_metrics.jsonl"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics.write_text("", encoding="utf-8")
    step = 0
    epoch = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=args.max_steps, desc="LoRA fine-tuning", disable=rank != 0)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
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
                mean_loss = loss.detach().float() / world_size
                if world_size > 1:
                    dist.all_reduce(mean_loss)
                raw_loss = mean_loss.item()
                if not math.isfinite(raw_loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: {raw_loss}")
                if rank == 0:
                    with metrics.open("a", encoding="utf-8") as stream:
                        row = {"step": step, "optimizer_steps": optimizer_steps,
                               "loss": raw_loss, "device": str(loss.device), "world_size": world_size}
                        stream.write(json.dumps(row) + "\n")
                progress.set_postfix(loss=f"{raw_loss:.4f}")
            if rank == 0 and step % args.save_steps == 0:
                ckpt = out_dir / f"checkpoint-{step}"
                adapter.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
            if step >= args.max_steps:
                break
        epoch += 1
    progress.close()
    if rank == 0:
        adapter.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"Saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
