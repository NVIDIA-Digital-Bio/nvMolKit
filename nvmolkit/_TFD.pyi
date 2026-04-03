# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Tuple

import numpy as np
from rdkit.Chem import Mol

def GetTFDMatricesCpuBuffer(
    mols: List[Mol],
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
) -> List[np.ndarray]: ...
def GetTFDMatricesGpuBuffer(
    mols: List[Mol],
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
) -> Tuple[Any, List[int]]: ...
