#!/usr/bin/env python3
"""Prepare a small instruction tuning JSONL dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def text(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def convert(row: dict) -> dict | None:
    instruction = text(row, ["instruction", "question", "query", "prompt", "input"])
    answer = text(row, ["output", "answer", "response", "completion"])
    context = text(row, ["context"])
    if not instruction or not answer:
        return None
    if context:
        instruction = f"{instruction}\n\nContext:\n{context}"
    return {"instruction": instruction, "input": "", "output": answer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medalpaca/medical_meadow_medical_flashcards")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/medical_flashcards_lora.jsonl")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [item for item in (convert(row) for row in load_dataset(args.dataset, split=args.split)) if item]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_samples]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {out}")


if __name__ == "__main__":
    main()
