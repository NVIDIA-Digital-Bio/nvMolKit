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

from typing import Iterable, Optional

from rdkit.Chem import Mol

DEFAULT_FF_SEARCH_SPACE = {
    "batchSize": (32, 1500, "log"),
    "batchesPerGpu": (1, 8),
}


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
