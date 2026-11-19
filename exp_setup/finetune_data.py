"""Read and batch instruction examples for causal LoRA training."""

import json

import torch
from torch.utils.data import Dataset


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
