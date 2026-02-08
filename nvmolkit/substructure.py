# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU-accelerated substructure search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit.Chem import Mol

from nvmolkit._substructure import SubstructSearchConfig, countSubstructMatches, hasSubstructMatch
from nvmolkit._substructure import getSubstructMatches as _getSubstructMatches

__all__ = [
    "SubstructSearchConfig",
    "getSubstructMatches",
    "countSubstructMatches",
    "hasSubstructMatch",
]


@dataclass(frozen=True)
class _CsrMatchView:
    atom_indices: np.ndarray
    match_indptr: np.ndarray
    pair_indptr: np.ndarray
    shape: tuple[int, int]

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, target_idx: int) -> "_CsrTargetView":
        return _CsrTargetView(self, target_idx)

    def get_pair(self, target_idx: int, query_idx: int) -> list[np.ndarray]:
        num_targets, num_queries = self.shape
        if target_idx < 0:
            target_idx += num_targets
        if query_idx < 0:
            query_idx += num_queries
        if not (0 <= target_idx < num_targets and 0 <= query_idx < num_queries):
            raise IndexError("pair index out of range")

        pair_idx = target_idx * num_queries + query_idx
        m0 = int(self.pair_indptr[pair_idx])
        m1 = int(self.pair_indptr[pair_idx + 1])
        out: list[np.ndarray] = []
        for m in range(m0, m1):
            a0 = int(self.match_indptr[m])
            a1 = int(self.match_indptr[m + 1])
            out.append(self.atom_indices[a0:a1])
        return out


@dataclass(frozen=True)
class _CsrTargetView:
    parent: _CsrMatchView
    target_idx: int

    def __len__(self) -> int:
        return self.parent.shape[1]

    def __getitem__(self, query_idx: int) -> list[np.ndarray]:
        return self.parent.get_pair(self.target_idx, query_idx)


def getSubstructMatches(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig = SubstructSearchConfig(),
) -> _CsrMatchView:
    atom_indices, match_indptr, pair_indptr, shape = _getSubstructMatches(targets, queries, config)
    return _CsrMatchView(atom_indices, match_indptr, pair_indptr, shape)

