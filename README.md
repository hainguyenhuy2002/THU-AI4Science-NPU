# THU AI4Science NPU Instruction

Run from the repository root on a server with Ascend CANN and Ollama installed.
All examples run on one host. A multi-NPU run uses multiple cards on that host.

```bash
test -f .env || cp .env.example .env
set -a
source .env
set +a

npu-smi info
bash exp_setup/setup.sh deploy   # LLM deployment
bash exp_setup/setup.sh python   # Fine-tuning and training
```

Load `.env` in each terminal. Edit `NPU_DEVICES` there or override it on the
command line. Use physical card IDs from `npu-smi info`. Each process sees
logical device indices starting at zero for the selected cards.

Set `LLAMA_SERVER_BIN` or `PYTHON_BIN` in `.env` to reuse an existing runtime.
Use `HF_HUB_OFFLINE=1` if the training checkpoint is already cached on an offline
server. Sample inputs live in `exp_setup/`.

The multi-NPU Python commands start one PyTorch process per card with
`torchrun` and HCCL. The scripts default to `HCCL_NPU_SOCKET_PORT_RANGE=auto`
so CANN can allocate communication ports on a shared card.
[Ascend HCCL port configuration](https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0144.html)

To launch your own two-card Python file, load CANN and use the same launcher.

```bash
export NPU_DEVICES=0,1 HCCL_NPU_SOCKET_PORT_RANGE=auto
source exp_setup/ascend_env.sh
"${PYTHON_BIN:-.venv/bin/python}" -m torch.distributed.run \
  --standalone --nproc-per-node=2 your_script.py
```

## 1. LLM Deployment

### Sample Run

If Ollama is stopped, start its daemon in a separate terminal.

```bash
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" ollama serve
```

#### Single NPU

```bash
NPU_DEVICES=0 bash scripts/deploy/deploy.sh
```

#### Multiple NPUs

```bash
NPU_DEVICES=0,1 bash scripts/deploy/deploy.sh
```

Run one server at a time on the configured port. From another terminal, send
the sample request and check NPU usage.

```bash
curl --noproxy '*' -fsS "http://127.0.0.1:$LLAMA_CPP_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @exp_setup/chat.json
npu-smi info
```

Check for the server PID and a nonzero CANN model buffer on each selected
card. The example uses the Q8_0 Ollama tag in `.env`.
[CANN backend guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)

### Sample code

Both variants get the model file from the local Ollama daemon.

```bash
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
ollama show "$OLLAMA_MODEL" >/dev/null 2>&1 || ollama pull "$OLLAMA_MODEL"
gguf=$(ollama show "$OLLAMA_MODEL" --modelfile | sed -n 's/^FROM //p')
server="${LLAMA_SERVER_BIN:-llama.cpp/build/bin/llama-server}"
```

#### Single NPU

```bash
export NPU_DEVICES=0
source exp_setup/ascend_env.sh
"$server" --model "$gguf" --alias "$MODEL_ALIAS" --host 127.0.0.1 \
  --port "$LLAMA_CPP_PORT" --n-gpu-layers all --split-mode none \
  --parallel 1 --threads "${THREADS:-8}" -lv 4
```

#### Multiple NPUs

Split the layers across two cards with equal placement weights.

```bash
export NPU_DEVICES=0,1
source exp_setup/ascend_env.sh
"$server" --model "$gguf" --alias "$MODEL_ALIAS" --host 127.0.0.1 \
  --port "$LLAMA_CPP_PORT" --n-gpu-layers all --split-mode layer --tensor-split 1,1 \
  --parallel 1 --threads "${THREADS:-8}" -lv 4
```

## 2. Fine-tuning

### Sample run  

`BASE_MODEL` selects a Transformers checkpoint or a local checkpoint directory.
The example fine-tunes Qwen 2.5 0.5B Instruct using LoRA. Each DDP worker keeps a
full copy of the base model and trains on a share of the examples.

#### Single NPU

```bash
NPU_DEVICES=0 bash scripts/finetuning/finetune.sh
```

#### Multiple NPUs

```bash
NPU_DEVICES=0,1 bash scripts/finetuning/finetune.sh
```

The launcher prints the device and PID of each worker. Check those PIDs in
`npu-smi info` during training. Look for two optimizer steps and a `world_size`
of 1 or 2 in the saved `training_metrics.jsonl`. Rank 0 saves the adapter.

### Sample code

Start with a Transformers `model`. The single-card example takes a tokenized
`batch` with `labels`. The multi-card example takes a `dataset` of tokenized
examples with `labels`.

#### Single NPU

```python
import torch
import torch_npu
from peft import LoraConfig, get_peft_model

device = torch.device("npu:0")
config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, config).to(device).train()
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)

batch = {name: tensor.to(device) for name, tensor in batch.items()}
optimizer.zero_grad()
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

#### Multiple NPUs

```python
import os

import torch
import torch.distributed as dist
import torch_npu
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("npu", local_rank)
torch.npu.set_device(device)
dist.init_process_group("hccl")

config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, config).to(device)
model = DDP(model, device_ids=[local_rank], broadcast_buffers=False).train()
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)

batch = next(iter(DataLoader(dataset, batch_size=1, sampler=DistributedSampler(dataset))))
batch = {name: tensor.to(device) for name, tensor in batch.items()}
optimizer.zero_grad()
loss = model(**batch).loss
loss.backward()
optimizer.step()
dist.destroy_process_group()
```

## 3. Training

### Sample run

The examples train a graph convolutional network on a synthetic graph. With
DDP, each worker uses the same graph and computes loss for a different subset
of labeled training nodes.

#### Single NPU

```bash
NPU_DEVICES=0 bash scripts/training/train.sh
```

#### Multiple NPUs

```bash
NPU_DEVICES=0,1 bash scripts/training/train.sh
```

Match the worker PIDs to `npu-smi info`. The final row in `gcn_metrics.jsonl`
should show epoch 20 and a `world_size` of 1 or 2. The output path appears at
the end of each run.

### Sample code

#### Single NPU

```python
import torch
import torch_npu
from exp_setup.graph_data import make_synthetic_graph
from scripts.training.train import GCN

device = torch.device("npu:0")
x, adj, labels, train_ids, _, _ = make_synthetic_graph(1200, 128, 7, 8, 7)
x, adj, labels, train_ids = (tensor.to(device) for tensor in (x, adj, labels, train_ids))

model = GCN(128, 128, 7, dropout=0.35).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss = torch.nn.functional.cross_entropy(model(x, adj)[train_ids], labels[train_ids])
loss.backward()
optimizer.step()
```

#### Multiple NPUs

```python
import os

import torch
import torch.distributed as dist
import torch_npu
from torch.nn.parallel import DistributedDataParallel as DDP
from exp_setup.graph_data import make_synthetic_graph
from scripts.training.train import GCN

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("npu", local_rank)
torch.npu.set_device(device)
dist.init_process_group("hccl")

x, adj, labels, train_ids, _, _ = make_synthetic_graph(1200, 128, 7, 8, 7)
rank, world_size = dist.get_rank(), dist.get_world_size()
local_ids = train_ids[rank::world_size].to(device)
x, adj, labels = (tensor.to(device) for tensor in (x, adj, labels))

model = GCN(128, 128, 7, dropout=0.35).to(device)
model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss = torch.nn.functional.cross_entropy(
    model(x, adj)[local_ids], labels[local_ids], reduction="sum"
) * (world_size / train_ids.numel())
loss.backward()
optimizer.step()
dist.destroy_process_group()
```
