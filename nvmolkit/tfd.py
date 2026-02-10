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

"""GPU-accelerated Torsion Fingerprint Deviation (TFD) calculation.

This module provides GPU-accelerated implementations of TFD calculation,
compatible with RDKit's TorsionFingerprints API.

TFD (Torsion Fingerprint Deviation) is a measure of conformational similarity
based on the comparison of torsion angles between conformers. It was described
in: Schulz-Gasch et al., JCIM, 52, 1499-1512 (2012).

Example usage:
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> import nvmolkit.tfd as tfd
    >>>
    >>> # Create molecule with conformers
    >>> mol = Chem.MolFromSmiles('CCCCC')
    >>> AllChem.EmbedMultipleConfs(mol, numConfs=5)
    >>>
    >>> # Compute TFD matrix (GPU-accelerated)
    >>> tfd_matrix = tfd.GetTFDMatrix(mol)
    >>>
    >>> # Batch processing
    >>> mols = [mol1, mol2, mol3]
    >>> tfd_matrices = tfd.GetTFDMatrices(mols)
"""

from typing import List

import torch

from nvmolkit import _TFD
from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit.types import AsyncGpuResult


def GetTFDMatrix(
    mol,
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
    backend: str = "gpu",
) -> List[float]:
    """Calculate the TFD matrix for conformers of a molecule.

    This function computes the pairwise Torsion Fingerprint Deviation (TFD)
    values between all conformers of a molecule. The result is a lower
    triangular matrix stored as a flat list.

    Args:
        mol: RDKit molecule with multiple conformers.
        useWeights: If True (default), use distance-based torsion weights.
        maxDev: Normalization mode for torsion deviations.
            'equal': All torsions normalized by 180 degrees (default).
            'spec': Each torsion uses its specific maximum deviation.
        symmRadius: Radius for Morgan fingerprint atom invariants used
            to detect symmetric atoms (default: 2).
        ignoreColinearBonds: If True (default), ignore single bonds
            adjacent to triple bonds.
        backend: Computation backend, 'gpu' (default) or 'cpu'.

    Returns:
        Lower triangular TFD matrix as a flat list. For C conformers,
        returns C*(C-1)/2 values in row-major order:
        [TFD(1,0), TFD(2,0), TFD(2,1), TFD(3,0), ...]

    Example:
        >>> mol = Chem.MolFromSmiles('CCCCC')
        >>> AllChem.EmbedMultipleConfs(mol, numConfs=4)
        >>> tfd_matrix = GetTFDMatrix(mol)
        >>> len(tfd_matrix)  # 4*(4-1)/2 = 6
        6
    """
    return list(
        _TFD.GetTFDMatrix(
            mol,
            useWeights=useWeights,
            maxDev=maxDev,
            symmRadius=symmRadius,
            ignoreColinearBonds=ignoreColinearBonds,
            backend=backend,
        )
    )


def GetTFDMatrices(
    mols: List,
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
    backend: str = "gpu",
) -> List[List[float]]:
    """Calculate TFD matrices for multiple molecules.

    Batch version of GetTFDMatrix that processes multiple molecules
    efficiently on the GPU.

    Args:
        mols: List of RDKit molecules, each with multiple conformers.
        useWeights: If True (default), use distance-based torsion weights.
        maxDev: Normalization mode ('equal' or 'spec').
        symmRadius: Radius for atom invariants (default: 2).
        ignoreColinearBonds: If True (default), ignore colinear bonds.
        backend: Computation backend, 'gpu' (default) or 'cpu'.

    Returns:
        List of TFD matrices, one per molecule. Each matrix is a flat
        list of lower triangular values.

    Example:
        >>> mols = [mol1, mol2, mol3]
        >>> tfd_matrices = GetTFDMatrices(mols)
        >>> len(tfd_matrices)
        3
    """
    results = _TFD.GetTFDMatrices(
        mols,
        useWeights=useWeights,
        maxDev=maxDev,
        symmRadius=symmRadius,
        ignoreColinearBonds=ignoreColinearBonds,
        backend=backend,
    )
    return [list(r) for r in results]


class TFDGpuResult:
    """Result container for GPU-resident TFD computation.

    This class holds GPU-resident TFD values along with metadata needed
    to extract per-molecule results. Use this when you want to keep
    results on GPU for further processing (e.g., clustering).

    Attributes:
        tfd_values: AsyncGpuResult containing all TFD values as a 1D tensor.
        output_starts: CSR-style boundaries for each molecule's TFD values.
        conformer_counts: Number of conformers per molecule.
    """

    def __init__(self, tfd_values: AsyncGpuResult, output_starts: List[int], conformer_counts: List[int]):
        self.tfd_values = tfd_values
        self.output_starts = output_starts
        self.conformer_counts = conformer_counts

    def extract_molecule(self, mol_idx: int) -> torch.Tensor:
        """Extract TFD matrix for a single molecule as a torch tensor.

        Args:
            mol_idx: Index of the molecule in the batch.

        Returns:
            1D torch tensor containing the TFD values for the molecule.
        """
        if mol_idx < 0 or mol_idx >= len(self.conformer_counts):
            raise IndexError(f"Molecule index {mol_idx} out of range [0, {len(self.conformer_counts)})")

        start = self.output_starts[mol_idx]
        end = self.output_starts[mol_idx + 1]
        return self.tfd_values.torch()[start:end]

    def extract_all(self) -> List[torch.Tensor]:
        """Extract TFD matrices for all molecules.

        Returns:
            List of 1D torch tensors, one per molecule.
        """
        all_values = self.tfd_values.torch()
        results = []
        for i in range(len(self.conformer_counts)):
            start = self.output_starts[i]
            end = self.output_starts[i + 1]
            results.append(all_values[start:end])
        return results

    def to_lists(self) -> List[List[float]]:
        """Convert GPU results to Python lists (blocking operation).

        Returns:
            List of TFD matrices as Python lists.
        """
        torch.cuda.synchronize()
        all_values = self.tfd_values.numpy()
        results = []
        for i in range(len(self.conformer_counts)):
            start = self.output_starts[i]
            end = self.output_starts[i + 1]
            results.append(all_values[start:end].tolist())
        return results


def GetTFDMatricesGpu(
    mols: List,
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
) -> TFDGpuResult:
    """Calculate TFD matrices with GPU-resident output.

    Similar to GetTFDMatrices but keeps results on GPU for further
    processing. Useful when chaining with other GPU operations like
    clustering.

    Args:
        mols: List of RDKit molecules, each with multiple conformers.
        useWeights: If True (default), use distance-based torsion weights.
        maxDev: Normalization mode ('equal' or 'spec').
        symmRadius: Radius for atom invariants (default: 2).
        ignoreColinearBonds: If True (default), ignore colinear bonds.

    Returns:
        TFDGpuResult containing GPU-resident TFD values and metadata.

    Example:
        >>> result = GetTFDMatricesGpu(mols)
        >>> # Access as torch tensor (async)
        >>> tensor = result.tfd_values.torch()
        >>> # Or extract per-molecule
        >>> mol0_tfd = result.extract_molecule(0)
    """
    pyarray, output_starts, conformer_counts = _TFD.GetTFDMatricesGpuBuffer(
        mols,
        useWeights=useWeights,
        maxDev=maxDev,
        symmRadius=symmRadius,
        ignoreColinearBonds=ignoreColinearBonds,
    )
    return TFDGpuResult(
        tfd_values=AsyncGpuResult(pyarray),
        output_starts=list(output_starts),
        conformer_counts=list(conformer_counts),
    )


def GetTFDMatrixGpu(
    mol,
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
) -> AsyncGpuResult:
    """Calculate TFD matrix for a single molecule with GPU-resident output.

    Args:
        mol: RDKit molecule with multiple conformers.
        useWeights: If True (default), use distance-based torsion weights.
        maxDev: Normalization mode ('equal' or 'spec').
        symmRadius: Radius for atom invariants (default: 2).
        ignoreColinearBonds: If True (default), ignore colinear bonds.

    Returns:
        AsyncGpuResult containing TFD values as a 1D tensor.
    """
    result = GetTFDMatricesGpu(
        [mol],
        useWeights=useWeights,
        maxDev=maxDev,
        symmRadius=symmRadius,
        ignoreColinearBonds=ignoreColinearBonds,
    )
    return result.tfd_values
