# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Tuple

from rdkit.Chem import Mol

def GetTFDMatrix(
    mol: Mol,
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
    backend: str = "gpu",
) -> List[float]: ...
def GetTFDMatrices(
    mols: List[Mol],
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
    backend: str = "gpu",
) -> List[List[float]]: ...
def GetTFDMatricesGpuBuffer(
    mols: List[Mol],
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
) -> Tuple[Any, List[int], List[int]]: ...
