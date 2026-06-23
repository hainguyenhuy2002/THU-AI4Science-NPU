#!/usr/bin/env python3
"""Print the local GGUF blob path used by an Ollama model."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Ollama model name, for example llama3.3:70b")
    parser.add_argument("--ollama-bin", default=os.getenv("OLLAMA_BIN"))
    args = parser.parse_args()
    ollama_bin = args.ollama_bin or shutil.which("ollama") or str(Path.home() / ".local/bin/ollama")
    if not Path(ollama_bin).exists():
        raise SystemExit("Could not find ollama. Set OLLAMA_BIN=/path/to/ollama.")

    result = subprocess.run(
        [ollama_bin, "show", args.model, "--modelfile"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            result.stderr.strip()
            or f"ollama show failed for {args.model}. Start Ollama with: {ollama_bin} serve"
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
