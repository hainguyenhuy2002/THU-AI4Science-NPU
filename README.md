# THU AI4Science NPU Instruction

Run these commands from the repository root on a server with Ascend CANN and
Ollama already installed. Check available device memory with `npu-smi info`.

```bash
cp .env.example .env
# Edit .env to choose your model and physical NPU.
set -a
source .env
set +a
```

`OLLAMA_MODEL` is the deployment tag. `BASE_MODEL` is the matching Transformers
checkpoint or a local checkpoint directory for fine-tuning. The example uses
Qwen 2.5 0.5B Instruct. Set both Ascend visibility variables to the physical
card you want to use. PyTorch calls the first visible card `npu:0`.

## 1. LLM Deployment

Ollama provides the GGUF weights. The CANN build of llama.cpp serves the
model on Ascend. Choose a model architecture and weight format supported by
your build. Start with a CANN-compatible tag such as Q8_0 when available.
[llama.cpp CANN guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)

```bash
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
# If the Ollama daemon is stopped, run ollama serve with this host in another terminal.
ollama show "$OLLAMA_MODEL" || ollama pull "$OLLAMA_MODEL"

# Run the setup once, or set LLAMA_SERVER_BIN to an existing CANN build.
bash scripts/llm_deploy/setup_llama_cpp_cann.sh
bash scripts/llm_deploy/serve_ollama_model_on_npu.sh
curl -fsS "http://127.0.0.1:$LLAMA_CPP_PORT/health"
```

Verify that the server appears in `npu-smi info` and that
`logs/llm_deploy.log` shows a CANN model buffer. A successful HTTP response
only confirms that the server is ready. If the model buffer is mostly on the
CPU, choose another GGUF variant supported by CANN.

Use the OpenAI-compatible chat endpoint from your application.

```python
import json
import os
from urllib.request import Request, urlopen

url = f"http://127.0.0.1:{os.environ['LLAMA_CPP_PORT']}/v1/chat/completions"
payload = {
    "model": os.environ["MODEL_ALIAS"],
    "messages": [{"role": "user", "content": "What is 2 plus 2?"}],
    "max_tokens": 64,
}
request = Request(url, data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=30) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

Stop the example server with `kill "$(cat logs/llama-server.pid)"` when finished.

## 2. Fine-tuning

Choose a Transformers checkpoint with a chat template and PEFT support.
This example trains a LoRA adapter. The frozen base checkpoint still needs
to fit on the NPU. Set up the environment once, or reuse an existing
environment with PyTorch, torch-npu, Transformers, and PEFT.

```bash
bash scripts/finetune/setup_env.sh
source scripts/common/ascend_env.sh
source .venv-finetune/bin/activate

mkdir -p outputs
run_dir="$(mktemp -d "$PWD/outputs/lora-smoke.XXXXXX")"
cat > "$run_dir/train.jsonl" <<'JSONL'
{"instruction":"What is two plus two?","output":"4"}
{"instruction":"Name the capital of France.","output":"Paris"}
JSONL

python scripts/finetune/train_lora_npu.py \
  --model "$BASE_MODEL" --data "$run_dir/train.jsonl" \
  --output-dir "$run_dir/adapter" --system-prompt 'Answer concisely.' \
  --target-modules all-linear --max-steps 2 --grad-accum 1 --cutoff-len 128
```

Watch `npu-smi info` during training. After the run, check
`training_metrics.jsonl` and `adapter_model.safetensors` in
`$run_dir/adapter`. The final metric should show two optimizer steps on
`npu:0`. For your project, replace the JSONL with your data and adjust
the step count and sequence length. You can pass explicit layer names
after `--target-modules` when your model needs them.

To integrate the training step into your own code, start with your loaded
Transformers model and a tokenized batch containing `labels`.

```python
import torch
import torch_npu
from peft import LoraConfig, get_peft_model

device = torch.device("npu:0")
model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM",
                                       target_modules="all-linear")).to(device)
model.train()
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=2e-4
)

# Repeat for each batch.
batch = {name: tensor.to(device) for name, tensor in batch.items()}
optimizer.zero_grad()
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

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
