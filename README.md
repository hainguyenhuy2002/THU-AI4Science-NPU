# THU AI4Science NPU Instruction

Run from the repository root on a server with Ascend CANN and Ollama installed.
Check free accelerator memory with `npu-smi info`.

```bash
test -f .env || cp .env.example .env
# Edit .env to select your physical NPU, model, and local ports.
set -a
source .env
set +a
```

Load `.env` in each terminal you use. Set both Ascend visibility variables to
the same physical card. PyTorch calls the first visible card `npu:0`.

Run the setup for the tasks you need. Set `LLAMA_SERVER_BIN` or `PYTHON_BIN` in
`.env` to reuse an existing runtime. If your Transformers checkpoint is already
cached on an offline server, set `HF_HUB_OFFLINE=1`.

```bash
bash exp_setup/setup.sh deploy   # LLM deployment
bash exp_setup/setup.sh python   # Fine-tuning and training
```

`exp_setup/` contains the chat request, the instruction JSONL, and the synthetic
graph generator used in the examples.

## 1. LLM Deployment

### Run

Start Ollama in a separate terminal if its daemon is stopped. Leave it running.

```bash
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" ollama serve
```

```bash
bash scripts/deploy/deploy.sh
```

When the server is ready, send the example request from another terminal.

```bash
curl --noproxy '*' -fsS "http://127.0.0.1:$LLAMA_CPP_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @exp_setup/chat.json
npu-smi info
```

Look for a `llama-server` process on the selected NPU and a CANN model buffer
in the server log. Stop the server with Ctrl+C when finished.

### NPU code

Ollama provides the GGUF file for the tag in `OLLAMA_MODEL`. The CANN build
of llama.cpp serves it on Ascend. Choose a format supported by your build,
such as Q8_0. [CANN backend guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)

```bash
source exp_setup/ascend_env.sh
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
ollama show "$OLLAMA_MODEL" >/dev/null 2>&1 || ollama pull "$OLLAMA_MODEL"
gguf=$(ollama show "$OLLAMA_MODEL" --modelfile | sed -n 's/^FROM //p')

"${LLAMA_SERVER_BIN:-llama.cpp/build/bin/llama-server}" \
  --model "$gguf" --alias "$MODEL_ALIAS" --host 127.0.0.1 --port "$LLAMA_CPP_PORT" \
  --n-gpu-layers all --ctx-size 4096 --parallel 1 --threads "${THREADS:-8}" -lv 4
```

## 2. Fine-tuning

### Run

`BASE_MODEL` in `.env` selects a Transformers checkpoint or a local checkpoint
directory. It needs a chat template, PEFT support, and enough free NPU memory.
The example uses Qwen 2.5 0.5B Instruct.

```bash
bash scripts/finetuning/finetune.sh
```

Run `npu-smi info` during training. The script writes the adapter and
`training_metrics.jsonl` to a new directory under `outputs/`. The final metric
should show two optimizer steps on `npu:0`. Set `DATA_PATH` to use your own
instruction JSONL and `MAX_STEPS` to change the number of steps.

### NPU code

With CANN loaded, start with your Transformers model and a tokenized
`batch` containing `labels`.

```python
import torch
import torch_npu
from peft import LoraConfig, get_peft_model

device = torch.device("npu:0")
config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, config).to(device)
model.train()
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=2e-4
)

batch = {name: tensor.to(device) for name, tensor in batch.items()}
optimizer.zero_grad(set_to_none=True)
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

## 3. Training

### Run

The example trains a graph convolutional network on a synthetic graph.

```bash
bash scripts/training/train.sh
```

Run `npu-smi info` during training. The script writes `gcn_metrics.jsonl` to a
new directory under `outputs/`. The final metric should show epoch 20 on
`npu:0`. Set `EPOCHS` or `NODES` to change the run.

### NPU code

Load CANN before running Python. Move the graph tensors and model to the same
device before the forward and backward passes.

```python
import torch
import torch_npu
from exp_setup.graph_data import make_synthetic_graph
from scripts.training.train import GCN

device = torch.device("npu:0")
x, adj, labels, train_ids, _, _ = make_synthetic_graph(1200, 128, 7, 8, 7)
x, adj = x.to(device), adj.to(device)
labels, train_ids = labels.to(device), train_ids.to(device)

model = GCN(128, 128, 7, dropout=0.35).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
optimizer.zero_grad(set_to_none=True)
logits = model(x, adj)
loss = torch.nn.functional.cross_entropy(logits[train_ids], labels[train_ids])
loss.backward()
optimizer.step()
```
