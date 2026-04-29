# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for forcefield-optimize autotuners.

The MMFF and UFF batched optimizers share the same hardware knobs: the user
controls only ``batchSize`` and ``batchesPerGpu``; ``preprocessingThreads`` is
not consumed by the C++ batched-FF code path (threads come from
``batchesPerGpu * numGpus``).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from rdkit.Chem import Mol


def clone_with_confs(mols: list[Mol]) -> list[Mol]:
    """Return deep copies of ``mols`` preserving their conformers.

    Used so that per-trial in-place coordinate updates do not contaminate the
    user's input molecules across trials.
    """
    return [Mol(mol) for mol in mols]


def total_conformers(mols: list[Mol]) -> int:
    """Return the sum of conformer counts across ``mols``."""
    return sum(mol.GetNumConformers() for mol in mols)


def coerce_gpu_ids(gpuIds: Optional[Iterable[int]]) -> list[int]:
    """Normalize a user-provided GPU id selection to a fixed list."""
    return list(gpuIds) if gpuIds is not None else []


def cpu_count() -> int:
    """Return the usable CPU core count, with a floor of 1."""
    return max(1, os.cpu_count() or 1)


def resolve_cpu_budget(cpu_budget: Optional[int]) -> int:
    """Return the effective CPU budget for autotune search ranges.

    ``cpu_budget`` lets callers cap total CPU usage explicitly (useful for
    cross-machine normalization where the goal is to isolate GPU performance
    from CPU-count differences). When ``None``, falls back to
    :func:`cpu_count`. Values less than 1 are rejected.
    """
    if cpu_budget is None:
        return cpu_count()
    if cpu_budget < 1:
        raise ValueError(f"cpu_budget must be >= 1, got {cpu_budget}")
    return int(cpu_budget)


def resolve_num_gpus(fixed_gpu_ids: list[int]) -> int:
    """Resolve the GPU count the runtime will actually use.

    When the caller fixed an explicit list of GPU IDs, that count wins.
    Otherwise we query CUDA via ``torch.cuda.device_count`` (matching the
    runtime's "all available GPUs" default). Returns at least 1 so callers
    can safely use the value as a divisor.
    """
    if fixed_gpu_ids:
        return max(1, len(fixed_gpu_ids))
    try:
        import torch  # noqa: PLC0415

        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


def default_ff_search_space(num_gpus: int, cpus: int) -> dict:
    """Return the default FF search space scaled to the active hardware.

    ``batchesPerGpu`` is per-GPU; its upper bound is capped at
    ``cpus // num_gpus`` so the total worker thread count across all GPUs
    (``num_gpus * batchesPerGpu``) does not exceed ``cpus``.
    """
    per_gpu_max = max(1, cpus // max(1, num_gpus))
    return {
        "batchSize": (32, 1500, "log"),
        "batchesPerGpu": (1, per_gpu_max),
    }
