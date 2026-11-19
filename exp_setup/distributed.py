"""Select one NPU per process and initialize HCCL for torchrun workers."""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


@contextmanager
def npu_run():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank >= torch.npu.device_count():
        raise RuntimeError(f"NPU {local_rank} is not visible. Check NPU_DEVICES.")
    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    if world_size > 1:
        dist.init_process_group("hccl")
    try:
        backend = dist.get_backend() if dist.is_initialized() else "single"
        print(f"rank={rank} world_size={world_size} device={device} backend={backend} pid={os.getpid()}", flush=True)
        yield device, rank, world_size
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
