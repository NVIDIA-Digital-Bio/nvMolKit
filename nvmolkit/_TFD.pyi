# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from rdkit.Chem import Mol

def GetTFDMatricesCpuBuffer(
    mols: list[Mol],
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
) -> list[np.ndarray]: ...
def GetTFDMatricesGpuBuffer(
    mols: list[Mol],
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
) -> tuple[Any, list[int]]: ...
