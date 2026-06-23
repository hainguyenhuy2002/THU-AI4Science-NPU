#!/usr/bin/env python3
"""Print the local GGUF blob path used by an Ollama model."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Ollama model name, for example llama3.3:70b")
    args = parser.parse_args()

    result = subprocess.run(
        ["ollama", "show", args.model, "--modelfile"],
        check=True,
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines():
        match = re.match(r"^FROM\s+(.+)$", line.strip())
        if not match:
            continue
        path = os.path.expanduser(match.group(1))
        if Path(path).is_file():
            print(path)
            return
    raise SystemExit(f"Could not find a local FROM blob for {args.model}. Run: ollama pull {args.model}")


if __name__ == "__main__":
    main()
