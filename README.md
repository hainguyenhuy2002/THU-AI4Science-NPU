# THU AI4Science NPU Instruction

Run the commands below from the repository root on the Ascend server. Run
`npu-smi info` first and choose an available device. The examples select physical
NPU 0. Change both visibility variables to use a different card.

## 1. LLM Deployment

Ollama downloads and stores the model. A CANN build of `llama.cpp` serves the
weights and handles inference on Ascend. The `mistral-small:latest` tag used here
is Mistral Small 3 24B Instruct (2501), with Q4_K_M weights. On the tested CANN
build, Q4_K_M placed almost all model weights on the CPU. Use Q8_0 for this
deployment. See the [Ollama model tags](https://ollama.com/library/mistral-small/tags)
and [llama.cpp CANN build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md).

### 1.1 Install and start Ollama

If Ollama is already installed, add its binary directory to `PATH` and skip the
download. For a user-local installation on Linux ARM64, use `curl`, `tar`, and
`zstd` with the commands below. On x86-64, use the `amd64` archive. See the
[official Linux instructions](https://docs.ollama.com/linux) for system installs.

```bash
mkdir -p "$HOME/.local"
curl -fL --retry 3 https://ollama.com/download/ollama-linux-arm64.tar.zst \
  | tar --zstd -xf - -C "$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
ollama --version
```

If an Ollama daemon is already listening on port 11434, reuse it. Otherwise,
start the following in a separate terminal and leave it running. Use the same
`OLLAMA_MODELS` directory for the daemon and the server-side GGUF lookup.

```bash
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$HOME/.ollama/models"
ollama serve
```

Back in the repository terminal, pull the model if it is missing and record the
local GGUF path. Keeping the original blob also lets you repeat the cached-weight
fine-tuning test in Part 2.

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama show mistral-small:latest || ollama pull mistral-small:latest
ollama show mistral-small:latest
mistral_gguf="$(python3 scripts/llm_deploy/find_ollama_gguf.py mistral-small:latest)"
```

The model should report 23.6B parameters and Q4_K_M quantization. If `latest`
changes, use `mistral-small:24b-instruct-2501-q4_K_M` to select the tag used here.

### 1.2 Build llama.cpp and prepare Q8_0 weights

With a working Ascend driver and CANN toolkit, install CMake and a C/C++ compiler
if needed. Build once per checkout. The recheck used CANN 8.5.0 and llama.cpp
commit `ac4105d68b2955027115cf9bb50941ccf56974eb`. When reusing a build, set
`llama_bin` to its absolute `build/bin` directory. If GitHub is unreachable from
the server, set `LLAMA_CPP_REPO` to a reachable mirror or a local clone that
contains the pinned commit.

```bash
export ASCEND_VISIBLE_DEVICES=0 ASCEND_RT_VISIBLE_DEVICES=0
LLAMA_CPP_REF=ac4105d68b2955027115cf9bb50941ccf56974eb BUILD_JOBS=16 \
  bash scripts/llm_deploy/setup_llama_cpp_cann.sh
source scripts/common/ascend_env.sh
llama_bin="$PWD/llama.cpp/build/bin"
"$llama_bin/llama-server" --list-devices
```

`--list-devices` should show a `CANN0` device. The tested path creates a
separate Q8_0 file from the cached Ollama blob. Allow about 24 GiB of additional
disk space. Requantization keeps the rounding from Q4_K_M and can add further
rounding. The direct Q8_0 download below avoids that extra conversion.

```bash
mkdir -p outputs/models
q8_gguf="$PWD/outputs/models/mistral-small-2501-Q8_0.gguf"
"$llama_bin/llama-quantize" --allow-requantize "$mistral_gguf" "$q8_gguf" Q8_0 16
```

Ollama also publishes Q8_0 directly. To use that variant, run
`ollama pull mistral-small:24b-instruct-2501-q8_0` and set `q8_gguf` to the output
of `python3 scripts/llm_deploy/find_ollama_gguf.py mistral-small:24b-instruct-2501-q8_0`.
The cluster47 test used the local requantization above.

### 1.3 Serve the model and verify NPU inference

The API listens on localhost. Set `LLAMA_CPP_PORT` to a free port if 18080 is
occupied. The smoke test uses a 1,024-token context and one request slot.

```bash
MODEL_PATH="$q8_gguf" \
LLAMA_SERVER_BIN="$llama_bin/llama-server" \
MODEL_ALIAS=mistral-small-ascend \
LLAMA_CPP_PORT=18080 CTX_SIZE=1024 THREADS=8 N_GPU_LAYERS=all \
LOG_DIR="$PWD/logs/mistral-deploy" \
LLAMA_SERVER_EXTRA_ARGS='--no-mmap --fit off --parallel 1 -lv 4' \
bash scripts/llm_deploy/serve_ollama_model_on_npu.sh
```

```bash
curl --noproxy '*' -fsS http://127.0.0.1:18080/health
curl --noproxy '*' -fsS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral-small-ascend","messages":[{"role":"user","content":"What is 2 plus 2? Reply with just the number."}],"temperature":0,"max_tokens":12}'
grep -E 'offloaded|CANN[0-9]+ model buffer|CPU(_REPACK)? model buffer' logs/mistral-deploy/llm_deploy.log
npu-smi info
```

Expect health `ok`, a chat response of `4`, a CANN model buffer of roughly
23,200 MiB, and a `llama-server` process using roughly 24,000 MB on the selected
NPU. Check **model buffer placement** even if the log says all layers were
offloaded. On cluster47 the Q4_K_M file produced the same answer but used only
about 1.6 MiB of CANN model buffers.

When the test is done, stop this server before reusing the card for training.

```bash
kill "$(cat logs/mistral-deploy/llama-server.pid)"
```

## 2. Fine-tuning

Train a LoRA adapter for **Mistral Small 3 24B Instruct (2501)** using
Transformers, PEFT, and `torch-npu`. Use the original BF16 Mistral checkpoint
when possible. If you already have the Ollama download, you can also load its
GGUF weights through Transformers and dequantize them for training. Both
options use the model's own tokenizer and chat template.

The two-step smoke test is sized for one 64 GiB Ascend 910B3 with short inputs.
The base weights alone take about 44 GiB in BF16. Start with an idle card and
at least 50 GiB of free HBM for weights, adapters, and training memory. Check
`HBM-Usage(MB)` in `npu-smi info` before starting. Stop any inference process
you started on that card. The GGUF route also needs substantial host RAM for
temporary dequantization. Allow about 160 GiB of available host RAM and check
`free -h`.

On cluster47 the cached GGUF loaded all 363 tensors. The run reached the NPU
transfer, where it ran out of HBM after another job occupied all eight devices.
The Mistral optimizer steps and saved adapter still need an end-to-end run on
an idle card.

### 2.1 Set up PyTorch and torch-npu

Run the setup script once. The cluster47 environment used Python 3.10,
PyTorch 2.5.1, `torch-npu` 2.5.1.post1, and Transformers 4.57.6.
Match the Torch and torch-npu wheel pair to the site's installed CANN toolkit.

```bash
export ASCEND_VISIBLE_DEVICES=0 ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/finetune/setup_env.sh
source scripts/common/ascend_env.sh
source .venv-finetune/bin/activate
```

The setup check should print `npu available: True`. If it fails to find a
CANN library, source the site's CANN environment before importing PyTorch.

### 2.2 Prepare the checkpoint and two example records

**Option A. Original BF16 weights.** Use the exact Mistral 2501 model revision.
This downloads about 44 GiB of weights. You can set `HF_ENDPOINT` to a reachable
mirror that hosts this revision if the server cannot reach Hugging Face directly.
The default endpoint is Hugging Face. On cluster47, the tokenizer and config
download worked with `HF_ENDPOINT=https://hf-mirror.com`.
[Mistral checkpoint](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)

```bash
mistral_id=mistralai/Mistral-Small-24B-Instruct-2501
mistral_revision=9527884be6e5616bdd54de542f9ae13384489724
model_dir="$PWD/outputs/models/mistral-small-2501-bf16"
hf download "$mistral_id" --revision "$mistral_revision" --local-dir "$model_dir" \
  --include 'config.json' 'generation_config.json' 'model-*.safetensors' \
  'model.safetensors.index.json' 'tokenizer*' 'special_tokens_map.json' 'chat_template*'
weight_args=()
```

**Option B. Reuse the Ollama download.** Fetch only the matching tokenizer and
model config, then use the `mistral-small:latest` GGUF from Part 1. Transformers
dequantizes the stored weights before loading them in BF16 on the NPU.

```bash
mistral_id=mistralai/Mistral-Small-24B-Instruct-2501
mistral_revision=9527884be6e5616bdd54de542f9ae13384489724
model_dir="$PWD/outputs/models/mistral-small-2501-config"
python -m pip install 'gguf==0.19.0'
hf download "$mistral_id" --revision "$mistral_revision" --local-dir "$model_dir" \
  --include 'config.json' 'generation_config.json' 'tokenizer*' \
  'special_tokens_map.json' 'chat_template*'
mistral_gguf="$(python scripts/llm_deploy/find_ollama_gguf.py mistral-small:latest)"
weight_args=(--gguf-file "$mistral_gguf")
```

Use a new run directory. Each JSONL line has an `instruction`, optional
`input`, and expected `output`.

```bash
mkdir -p outputs
run_dir="$(mktemp -d "$PWD/outputs/mistral-lora-smoke.XXXXXX")"
cat > "$run_dir/train.jsonl" <<'JSONL'
{"instruction":"What is two plus two? Reply with the number.","input":"","output":"4"}
{"instruction":"Give the capital of France.","input":"","output":"Paris."}
JSONL
```

### 2.3 Run LoRA and verify the saved adapter

```bash
set -o pipefail
python -u scripts/finetune/train_lora_npu.py \
  --model "$model_dir" "${weight_args[@]}" \
  --data "$run_dir/train.jsonl" --output-dir "$run_dir/adapter" \
  --system-prompt 'Answer concisely.' \
  --max-steps 2 --grad-accum 1 --cutoff-len 128 \
  2>&1 | tee "$run_dir/train.log"
```

In another terminal, run `npu-smi info` during training and look for the Python
PID on the selected physical NPU. A single visible physical device is
`npu:0` inside PyTorch. Check that the log reports this device and that the
final metric includes two optimizer updates and a finite loss.

```bash
tail -n 2 "$run_dir/adapter/training_metrics.jsonl"
ls -lh "$run_dir/adapter/adapter_model.safetensors" "$run_dir/adapter/adapter_config.json"
```

The output is a PEFT adapter. Keep it with the same checkpoint and tokenizer
used for training. For a real run, replace the sample JSONL, use a new run
directory, and increase `--max-steps` and `--cutoff-len` as needed. If an example's
answer is truncated, shorten the prompt or increase the cutoff. If weight
loading runs out of NPU memory, select a card with more free HBM. If a training
step runs out of memory after the weights load, keep the batch size at 1 and
reduce the sequence length.

## 3. Graph Model Training

Goal: train a graph neural network on Ascend NPU. The sample uses a simple dense GCN layer so the NPU logic is visible without extra graph libraries.

Abstract implementation:

```python
import torch
import torch.nn as nn

# GPU implementation:
# device = torch.device("cuda:0")

# NPU implementation:
import torch_npu
device = torch.device("npu:0")

x = x.to(device)
adj_norm = adj_norm.to(device)
labels = labels.to(device)
model = GCN(...).to(device)

logits = model(x, adj_norm)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

Diff / comparative logic:

```diff
  import torch
+ import torch_npu

- device = torch.device("cuda:0")
+ device = torch.device("npu:0")

- x = x.cuda()
- edge_index = edge_index.cuda()
- model = model.cuda()
+ x = x.to(device)
+ adj_norm = adj_norm.to(device)
+ model = model.to(device)

- nvidia-smi
+ npu-smi info
```

Sample command:

```bash
cd /villa/rhh25/THU-AI4Science-NPU
bash scripts/graph/setup_env.sh

ASCEND_VISIBLE_DEVICES=0 \
EPOCHS=200 \
NODES=2500 \
bash scripts/graph/run_gcn_training.sh
```

Quick smoke test:

```bash
ASCEND_VISIBLE_DEVICES=0 \
EPOCHS=20 \
NODES=1200 \
bash scripts/graph/run_gcn_training.sh
```

Verify:

```bash
npu-smi info
tail outputs/graph/gcn_metrics.jsonl
```

Expected NPU evidence: metrics include `"device": "npu:0"` and `python` appears in the NPU process table during training.
