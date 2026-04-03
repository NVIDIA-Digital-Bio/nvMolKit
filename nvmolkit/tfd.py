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
    >>> mol = Chem.MolFromSmiles('CCCCC')
    >>> AllChem.EmbedMultipleConfs(mol, numConfs=5)
    >>>
    >>> # Python lists (RDKit-compatible, default)
    >>> tfd_matrix = tfd.GetTFDMatrix(mol)
    >>>
    >>> # Numpy arrays (fast, CPU)
    >>> tfd_matrices = tfd.GetTFDMatrices(mols, return_type="numpy")
    >>>
    >>> # GPU tensors (fastest, no D2H copy)
    >>> tfd_matrices = tfd.GetTFDMatrices(mols, return_type="tensor")
"""

from contextlib import nullcontext
from typing import List, Union

import numpy as np
import torch

try:
    import nvtx as _nvtx

    def _nvtx_range(name, color="blue"):
        return _nvtx.annotate(name, color=color)
except ImportError:

    def _nvtx_range(name, color="blue"):
        return nullcontext()


from nvmolkit import _TFD
from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit.types import AsyncGpuResult


class _TFDGpuResult:
    """Internal result container for GPU-resident TFD computation."""

    def __init__(self, tfd_values: AsyncGpuResult, output_starts: List[int]):
        self.tfd_values = tfd_values
        self.output_starts = output_starts

    def to_tensors(self) -> List[torch.Tensor]:
        """Extract as list of GPU tensors (no D2H copy)."""
        n = len(self.output_starts) - 1
        all_values = self.tfd_values.torch()
        return [all_values[self.output_starts[i] : self.output_starts[i + 1]] for i in range(n)]

    def to_numpy(self) -> List[np.ndarray]:
        """Extract as list of numpy arrays (one bulk D2H copy)."""
        n = len(self.output_starts) - 1
        torch.cuda.synchronize()
        all_values = self.tfd_values.numpy()
        return [all_values[self.output_starts[i] : self.output_starts[i + 1]] for i in range(n)]

    def to_lists(self) -> List[List[float]]:
        """Extract as Python lists (bulk D2H + tolist)."""
        n = len(self.output_starts) - 1
        torch.cuda.synchronize()
        all_list = self.tfd_values.numpy().tolist()
        return [all_list[self.output_starts[i] : self.output_starts[i + 1]] for i in range(n)]


def _get_gpu_result(mols, useWeights, maxDev, symmRadius, ignoreColinearBonds):
    """Run GPU TFD computation and return _TFDGpuResult (no D2H copy)."""
    pyarray, output_starts = _TFD.GetTFDMatricesGpuBuffer(
        mols,
        useWeights=useWeights,
        maxDev=maxDev,
        symmRadius=symmRadius,
        ignoreColinearBonds=ignoreColinearBonds,
    )
    return _TFDGpuResult(
        tfd_values=AsyncGpuResult(pyarray),
        output_starts=output_starts,
    )


def _extract_gpu_result(gpu_result, return_type):
    """Extract results from _TFDGpuResult based on return_type."""
    if return_type == "tensor":
        return gpu_result.to_tensors()
    elif return_type == "numpy":
        return gpu_result.to_numpy()
    else:
        return gpu_result.to_lists()


def _extract_cpu_result(arrays, return_type):
    """Convert list of numpy arrays to requested return type."""
    if return_type == "numpy":
        return list(arrays)
    elif return_type == "tensor":
        return [torch.from_numpy(a) for a in arrays]
    else:
        return [a.tolist() for a in arrays]


def GetTFDMatrices(
    mols: List,
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
    backend: str = "gpu",
    return_type: str = "list",
) -> Union[List[List[float]], List[np.ndarray], List[torch.Tensor]]:
    """Calculate TFD matrices for multiple molecules.

    Args:
        mols: List of RDKit molecules, each with multiple conformers.
        useWeights: If True (default), use distance-based torsion weights.
        maxDev: Normalization mode ('equal' or 'spec').
        symmRadius: Radius for atom invariants (default: 2).
        ignoreColinearBonds: If True (default), ignore colinear bonds.
        backend: Computation backend, 'gpu' (default) or 'cpu'.
        return_type: Output format:
            'list' (default): List of Python lists (RDKit-compatible).
            'numpy': List of numpy arrays (float64 for CPU, float32 for GPU).
            'tensor': List of torch.Tensors (GPU tensors for GPU backend, CPU tensors for CPU backend).

    Returns:
        List of TFD matrices in the requested format.
    """
    if backend in ("gpu", "GPU"):
        gpu_result = _get_gpu_result(mols, useWeights, maxDev, symmRadius, ignoreColinearBonds)
        with _nvtx_range("GPU: split per-molecule results", color="yellow"):
            return _extract_gpu_result(gpu_result, return_type)

    if backend not in ("cpu", "CPU"):
        raise ValueError(f"backend must be 'gpu' or 'cpu', got: '{backend}'")

    arrays = _TFD.GetTFDMatricesCpuBuffer(
        mols,
        useWeights=useWeights,
        maxDev=maxDev,
        symmRadius=symmRadius,
        ignoreColinearBonds=ignoreColinearBonds,
    )
    with _nvtx_range("CPU: convert result format", color="gray"):
        return _extract_cpu_result(arrays, return_type)


def GetTFDMatrix(
    mol,
    useWeights: bool = True,
    maxDev: str = "equal",
    symmRadius: int = 2,
    ignoreColinearBonds: bool = True,
    backend: str = "gpu",
    return_type: str = "list",
) -> Union[List[float], np.ndarray, torch.Tensor]:
    """Calculate the TFD matrix for conformers of a molecule.

    Convenience wrapper over GetTFDMatrices for a single molecule.

    Args:
        mol: RDKit molecule with multiple conformers.
        useWeights: If True (default), use distance-based torsion weights.
        maxDev: Normalization mode ('equal' or 'spec').
        symmRadius: Radius for atom invariants (default: 2).
        ignoreColinearBonds: If True (default), ignore colinear bonds.
        backend: Computation backend, 'gpu' (default) or 'cpu'.
        return_type: Output format:
            'list' (default): Python list of floats (RDKit-compatible).
            'numpy': numpy float32 array (CPU).
            'tensor': GPU torch.Tensor (no D2H copy).

    Returns:
        Lower triangular TFD matrix as a flat list, numpy array, or GPU tensor.
    """
    results = GetTFDMatrices(
        [mol],
        useWeights=useWeights,
        maxDev=maxDev,
        symmRadius=symmRadius,
        ignoreColinearBonds=ignoreColinearBonds,
        backend=backend,
        return_type=return_type,
    )
    if not results:
        if return_type == "numpy":
            return np.array([], dtype=np.float32)
        elif return_type == "tensor":
            return torch.tensor([], dtype=torch.float32)
        return []
    return results[0]
