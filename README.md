# THU AI4Science NPU Examples

## 1. Ollama Deployment

Goal: use Ollama to manage/pull the model, then serve the GGUF model through `llama.cpp` compiled with Ascend CANN.

Abstract implementation:

```bash
# GPU deployment usually selects CUDA devices:
export CUDA_VISIBLE_DEVICES=0

# NPU deployment selects Ascend devices:
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU llama.cpp build:
# cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=on

# NPU llama.cpp build:
cmake -S llama.cpp -B llama.cpp/build -DGGML_CANN=on

# GPU server normally uses a CUDA llama-server binary.
# NPU server uses a CANN llama-server binary and the same GGUF model file.
llama.cpp/build/bin/llama-server \
  --host 0.0.0.0 \
  --port 18080 \
  --model /path/to/model.gguf \
  --alias model-ascend \
  --n-gpu-layers -1
```

Diff / comparative logic:

```diff
- export CUDA_VISIBLE_DEVICES=0
- cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=on
- nvidia-smi
+ export ASCEND_VISIBLE_DEVICES=0
+ export ASCEND_RT_VISIBLE_DEVICES=0
+ cmake -S llama.cpp -B llama.cpp/build -DGGML_CANN=on
+ npu-smi info
```


Sample command:

```bash
bash scripts/llm_deploy/setup_llama_cpp_cann.sh

ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OLLAMA_MODEL=llama3.3:70b \
MODEL_ALIAS=llama3.3-70b-ascend \
LLAMA_CPP_PORT=18080 \
LLAMA_SERVER_EXTRA_ARGS="--no-mmap" \
bash scripts/llm_deploy/serve_ollama_model_on_npu.sh
```

Verify:

```bash
curl http://localhost:18080/health
npu-smi info
```

Expected NPU evidence: `llama-server` appears in the NPU process table and HBM usage increases.

## 2. Finetuning

Goal: finetune a causal language model with LoRA on Ascend NPU using PyTorch + `torch-npu`.

Abstract implementation:

```python
import torch

# GPU implementation:
# device = torch.device("cuda:0")

# NPU implementation:
import torch_npu
device = torch.device("npu:0")

model = model.to(device)
batch = {key: value.to(device) for key, value in batch.items()}

outputs = model(**batch)
loss = outputs.loss
loss.backward()
optimizer.step()
```

Diff / comparative logic:

```diff
  import torch
+ import torch_npu

- device = torch.device("cuda:0")
+ device = torch.device("npu:0")

- CUDA_VISIBLE_DEVICES=0
- nvidia-smi
+ ASCEND_VISIBLE_DEVICES=0
+ ASCEND_RT_VISIBLE_DEVICES=0
+ npu-smi info
```


Sample command:

```bash
bash scripts/finetune/setup_env.sh

ASCEND_VISIBLE_DEVICES=0 \
MAX_SAMPLES=1000 \
MAX_STEPS=800 \
bash scripts/finetune/run_lora_finetune.sh
```

Quick smoke test:

```bash
ASCEND_VISIBLE_DEVICES=0 \
MAX_SAMPLES=50 \
MAX_STEPS=40 \
OUTPUT_DIR=outputs/finetune-smoke \
bash scripts/finetune/run_lora_finetune.sh
```

Verify:

```bash
npu-smi info
tail -f outputs/qwen2.5-0.5b-medical-lora/training_metrics.jsonl
```

Expected NPU evidence: `python` appears in the NPU process table and the training log prints `Using device: npu:0`.

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
