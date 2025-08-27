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

"""GPU-accelerated MMFF optimization for molecular conformers.

This module provides GPU-accelerated implementations of MMFF (Molecular Mechanics Force Field)
optimization for multiple molecules and conformers using CUDA and OpenMP.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol

from nvmolkit import _mmffOptimization
from nvmolkit.types import HardwareOptions


def MMFFOptimizeMoleculesConfs(
    molecules: list["Mol"],
    maxIters: int = 200,
    nonBondedThreshold: float = 100.0,
    hardwareOptions: HardwareOptions | None = None
) -> list[list[float]]:
    """Optimize conformers for multiple molecules using MMFF force field with BFGS minimization.
    
    This function performs GPU-accelerated MMFF optimization on multiple molecules with
    multiple conformers each. It uses CUDA for GPU acceleration and OpenMP for CPU
    parallelization to achieve high performance.
    
    Args:
        molecules: List of RDKit molecules to optimize. Each molecule should have
                  conformers already generated.
        maxIters: Maximum number of BFGS optimization iterations (default: 200)
        nonBondedThreshold: Radius threshold for non-bonded interactions in Ångströms (default: 100.0)
        numThreads: Number of OpenMP threads for parallel processing (default: 1)
        batchSize: Batch size for processing conformers, 0 for no batching (default: 0)
    
    Returns:
        List of lists of energies, where each inner list contains the optimized energies
        for all conformers of the corresponding molecule. The order matches the input
        molecule order and conformer iteration order.
        
    Raises:
        ValueError: If any molecule in the input list is invalid
        RuntimeError: If CUDA operations fail or optimization encounters errors
        
    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import rdDistGeom
        >>> import nvmolkit.mmff as mmff
        >>> 
        >>> # Load molecules and generate conformers
        >>> mol1 = Chem.MolFromSmiles('CCO')
        >>> mol2 = Chem.MolFromSmiles('CCC')
        >>> rdDistGeom.EmbedMultipleConfs(mol1, numConfs=5)
        >>> rdDistGeom.EmbedMultipleConfs(mol2, numConfs=3)
        >>> 
        >>> # Optimize with custom settings
        >>> energies = mmff.MMFFOptimizeMoleculesConfs(
        ...     [mol1, mol2], 
        ...     maxIters=500,
        ...     numThreads=4,
        ...     batchSize=32
        ... )
        >>> 
        >>> # energies[0] contains 5 energies for mol1's conformers
        >>> # energies[1] contains 3 energies for mol2's conformers
    
    Note:
        - Input molecules are modified in-place with optimized conformer coordinates
    """
    # Validate input
    if not molecules:
        return []
    
    for i, mol in enumerate(molecules):
        if mol is None:
            raise ValueError(f"Molecule at index {i} is None")
    
    # Call the C++ implementation
    if hardwareOptions is None:
        hardwareOptions = HardwareOptions()
    native_options = hardwareOptions._as_native()
    return _mmffOptimization.MMFFOptimizeMoleculesConfs(
        molecules,
        maxIters,
        nonBondedThreshold,
        native_options
    )
