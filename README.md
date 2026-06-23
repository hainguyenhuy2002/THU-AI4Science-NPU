# THU AI4Science NPU Examples

This repo shows how to use Ascend 910B3 NPU resources for three common AI workloads:

1. Deploying an LLM with an Ollama-pulled model
2. Fine-tuning an LLM with LoRA
3. Training a deep learning graph model

The examples were designed for `cluster47`, where `npu-smi` and CANN are already available.

## 0. Code Pattern: GPU to NPU

Most PyTorch training code only needs a small device-layer change.

CUDA GPU code usually looks like:

```python
import torch

device = torch.device("cuda:0")
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

Ascend NPU code should look like:

```python
import torch
import torch_npu

device = torch.device("npu:0")
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

Environment variables are also different:

```bash
# CUDA GPU
CUDA_VISIBLE_DEVICES=0
nvidia-smi

# Ascend NPU
ASCEND_VISIBLE_DEVICES=0
ASCEND_RT_VISIBLE_DEVICES=0
npu-smi info
```

Copyable examples are in:

```text
patterns/device_utils.py
patterns/cuda_to_npu_minimal.py
patterns/graph_model_npu_template.py
```

Run the minimal NPU example:

```bash
cd /villa/rhh25/THU-AI4Science-NPU
ASCEND_VISIBLE_DEVICES=0 STEPS=20 bash patterns/run_minimal_npu.sh
```

Run the graph template:

```bash
ASCEND_VISIBLE_DEVICES=0 EPOCHS=20 bash patterns/run_graph_template_npu.sh
```

## Quick Check

```bash
npu-smi info
```

Expected hardware:

```text
Name: 910B3
HBM-Usage(MB): ... / 65536
```

All scripts load the Ascend environment from `/usr/local/Ascend/.../set_env.sh` and use:

```bash
ASCEND_VISIBLE_DEVICES=0
ASCEND_RT_VISIBLE_DEVICES=0
```

Override those variables to select another NPU.

## 1. Deploying LLM with Ollama

Ollama is used to pull/manage the model. On Ascend NPU, the actual NPU serving path uses `llama.cpp` built with CANN, because standard Ollama may run on CPU only.

Flow:

```text
ollama pull -> locate Ollama GGUF blob -> llama.cpp CANN server -> OpenAI-compatible API
```

### Build llama.cpp with CANN

Run once:

```bash
cd /villa/rhh25/THU-AI4Science-NPU
bash scripts/llm_deploy/setup_llama_cpp_cann.sh
```

### Pull a Model with Ollama

Example:

```bash
export PATH="$HOME/.local/bin:$PATH"
ollama serve > logs/ollama.log 2>&1 &
ollama pull llama3.3:70b
ollama list
```

Find the GGUF blob path:

```bash
python3 scripts/llm_deploy/find_ollama_gguf.py llama3.3:70b
```

### Serve on NPU

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OLLAMA_MODEL=llama3.3:70b \
MODEL_ALIAS=llama3.3-70b-ascend \
LLAMA_SERVER_EXTRA_ARGS="--no-mmap" \
bash scripts/llm_deploy/serve_ollama_model_on_npu.sh
```

For a known local model file:

```bash
MODEL_PATH=/villa/rhh25/.ollama/models/blobs/sha256-... \
MODEL_ALIAS=my-model-ascend \
bash scripts/llm_deploy/serve_ollama_model_on_npu.sh
```

### Test the LLM API

```bash
MODEL_ALIAS=llama3.3-70b-ascend \
bash scripts/llm_deploy/test_llm_api.sh
```

### Verify NPU Usage

```bash
npu-smi info
```

Look for:

- `llama-server` in the process table
- increased `HBM-Usage(MB)`
- nonzero `AICore(%)` while generating

## 2. Fine-Tuning LLM on NPU

This example fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on a real medical instruction dataset:

```text
medalpaca/medical_meadow_medical_flashcards
```

The default job uses 1000 samples and 800 steps. On `cluster47`, this ran on NPU 0 and completed in a few minutes for the small Qwen model.

### Create Fine-Tuning Environment

```bash
cd /villa/rhh25/THU-AI4Science-NPU
bash scripts/finetune/setup_env.sh
```

The script installs:

- PyTorch CPU wheel
- `torch-npu`
- Transformers
- Datasets
- PEFT
- CANN Python helper dependencies

If direct Hugging Face access is blocked, the scripts default to:

```bash
HF_ENDPOINT=https://hf-mirror.com
```

### Run Fine-Tuning

Direct run:

```bash
ASCEND_VISIBLE_DEVICES=0 \
MAX_SAMPLES=1000 \
MAX_STEPS=800 \
bash scripts/finetune/run_lora_finetune.sh
```

Run inside `byobu`:

```bash
byobu new-session -d -s npu-finetune \
  'cd /villa/rhh25/THU-AI4Science-NPU && ASCEND_VISIBLE_DEVICES=0 MAX_SAMPLES=1000 MAX_STEPS=800 bash scripts/finetune/run_lora_finetune.sh 2>&1 | tee logs/finetune.log'
```

Attach:

```bash
byobu attach -t npu-finetune
```

Detach without stopping:

```text
Ctrl-a d
```

### Outputs

```text
data/medical_flashcards_lora.jsonl
outputs/qwen2.5-0.5b-medical-lora/
outputs/qwen2.5-0.5b-medical-lora/training_metrics.jsonl
logs/finetune.log
```

### Verify NPU Usage

```bash
npu-smi info
```

Look for:

- `python` in the NPU process table
- `HBM-Usage(MB)` increasing on NPU 0
- `AICore(%)` rising during active steps

Check training loss:

```bash
tail -f outputs/qwen2.5-0.5b-medical-lora/training_metrics.jsonl
```

## 3. Training Deep Learning Graph Models on NPU

This example trains a two-layer Graph Convolutional Network using only PyTorch tensor operations. It avoids extra graph libraries so it can run on a fresh Ascend environment.

The script creates a synthetic citation-like graph:

- nodes have class-dependent feature vectors
- edges connect mostly same-class nodes plus some noisy cross-class links
- the GCN learns node classification

### Create Graph Training Environment

```bash
cd /villa/rhh25/THU-AI4Science-NPU
bash scripts/graph/setup_env.sh
```

### Run GCN Training on NPU

```bash
ASCEND_VISIBLE_DEVICES=0 \
EPOCHS=200 \
NODES=2500 \
bash scripts/graph/run_gcn_training.sh
```

For a quick smoke test:

```bash
ASCEND_VISIBLE_DEVICES=0 \
EPOCHS=20 \
NODES=1200 \
bash scripts/graph/run_gcn_training.sh
```

### Outputs

```text
outputs/graph/gcn_metrics.jsonl
logs/graph_npu_before.log
logs/graph_npu_after.log
```

Inspect metrics:

```bash
tail outputs/graph/gcn_metrics.jsonl
```

### Verify NPU Usage

Run in another terminal while training:

```bash
npu-smi info
```

Look for:

- `python` in the NPU process table
- increased `HBM-Usage(MB)`
- nonzero `AICore(%)`

## Common Troubleshooting

### `torch_npu` is not available

Check:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

If this fails, verify that the `torch-npu` version matches the CANN version.

### HugePages is `0 / 0`

That is normal for these examples. It means hugepage-backed memory is not configured or not used by this runtime path.

### HBM is nonzero but AICore is 0

The process may be idle or between compute kernels. Send a request or watch during training steps:

```bash
watch -n 1 npu-smi info
```

### Stop Processes

```bash
pkill -f llama-server
pkill -f train_lora_npu.py
pkill -f train_gcn_npu.py
```
