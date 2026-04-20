# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed test for dtensor_full_tensor.

Run with torchrun (or torch.distributed.run):
    torchrun --nproc_per_node=2 tests/special_distributed/test_dtensor_full_tensor.py

dtensor_full_tensor is a drop-in replacement for DTensor.full_tensor() that
uses dist.all_gather instead of allgather_into_tensor_coalesced so that it
works on ROCm / RCCL backends.

This test verifies that the gathered result is numerically identical to the
original unsharded tensor for:
  - row-sharded (Shard(0)) tensors
  - column-sharded (Shard(1)) tensors
  - replicated tensors (no Shard placement)
  - bfloat16 dtype
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from verl.utils.fsdp_utils import dtensor_full_tensor


def _make_full_tensor(shape, dtype=torch.float32, seed=42):
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=dtype)


def test_row_sharded(mesh, device):
    """Shard(0): each rank holds a horizontal slice; gather must reconstruct rows."""
    full = _make_full_tensor((8, 4)).to(device)
    dtensor = DTensor.from_local(
        full.chunk(dist.get_world_size(), dim=0)[dist.get_rank()],
        mesh,
        [Shard(0)],
    )
    result = dtensor_full_tensor(dtensor, device)
    assert result.shape == full.shape, f"shape mismatch: {result.shape} vs {full.shape}"
    assert torch.allclose(result, full), "row-sharded gather produced wrong values"


def test_col_sharded(mesh, device):
    """Shard(1): each rank holds a vertical slice; gather must reconstruct cols."""
    full = _make_full_tensor((4, 8)).to(device)
    dtensor = DTensor.from_local(
        full.chunk(dist.get_world_size(), dim=1)[dist.get_rank()],
        mesh,
        [Shard(1)],
    )
    result = dtensor_full_tensor(dtensor, device)
    assert result.shape == full.shape, f"shape mismatch: {result.shape} vs {full.shape}"
    assert torch.allclose(result, full), "col-sharded gather produced wrong values"


def test_replicated(mesh, device):
    """Replicate: all ranks hold the same tensor; dtensor_full_tensor returns it unchanged."""
    full = _make_full_tensor((4, 4)).to(device)
    dtensor = DTensor.from_local(full, mesh, [Replicate()])
    result = dtensor_full_tensor(dtensor, device)
    assert result.shape == full.shape, f"shape mismatch: {result.shape} vs {full.shape}"
    assert torch.allclose(result, full), "replicated gather produced wrong values"


def test_bfloat16(mesh, device):
    """dtensor_full_tensor must preserve bfloat16 dtype."""
    full = _make_full_tensor((8, 4), dtype=torch.bfloat16).to(device)
    dtensor = DTensor.from_local(
        full.chunk(dist.get_world_size(), dim=0)[dist.get_rank()],
        mesh,
        [Shard(0)],
    )
    result = dtensor_full_tensor(dtensor, device)
    assert result.dtype == torch.bfloat16, f"dtype mismatch: {result.dtype}"
    assert torch.allclose(result, full), "bfloat16 gather produced wrong values"


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Use CPU so the test runs without a GPU / ROCm device.
    device = torch.device("cpu")
    mesh = init_device_mesh("cpu", (world_size,))

    tests = [test_row_sharded, test_col_sharded, test_replicated, test_bfloat16]
    for fn in tests:
        fn(mesh, device)
        if rank == 0:
            print(f"PASSED: {fn.__name__}")

    dist.destroy_process_group()
