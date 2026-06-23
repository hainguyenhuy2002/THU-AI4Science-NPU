# NPU Coding Patterns

These files show the small code changes needed to move student PyTorch code from CUDA GPU to Ascend NPU.

Use these examples as templates:

- `device_utils.py`: reusable device selection helpers
- `cuda_to_npu_minimal.py`: side-by-side GPU and NPU training loops
- `graph_model_npu_template.py`: graph-model training skeleton using NPU

The core idea:

```python
# CUDA GPU
device = torch.device("cuda:0")
model = model.to(device)
x = x.to(device)

# Ascend NPU
import torch_npu
device = torch.device("npu:0")
model = model.to(device)
x = x.to(device)
```

Use `npu-smi info` instead of `nvidia-smi` to verify usage.
