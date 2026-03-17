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

"""GPU-accelerated pairwise RMSD matrix for molecular conformers."""

from typing import TYPE_CHECKING

import torch

from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit import _conformerRmsd
from nvmolkit.types import AsyncGpuResult

if TYPE_CHECKING:
    from rdkit.Chem import Mol

__all__ = ["GetConformerRMSMatrix", "GetConformerRMSMatrixBatch"]


def GetConformerRMSMatrix(
    mol: "Mol",
    prealigned: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult:
    """Compute the pairwise RMSD matrix between all conformers of a molecule on GPU.

    This is a GPU-accelerated equivalent of RDKit's ``AllChem.GetConformerRMSMatrix``.
    For N conformers with M atoms, it computes N*(N-1)/2 pairwise RMSD values using
    one GPU thread-block per pair.  When ``prealigned`` is False (default), each pair
    is optimally superimposed via the Kabsch algorithm before computing RMSD.

    The result can be passed directly to :func:`nvmolkit.clustering.butina` for
    GPU-accelerated Butina clustering, keeping the entire conformer-selection
    pipeline on the GPU.

    Args:
        mol: RDKit molecule with two or more conformers.  Strip hydrogens first
             (``Chem.RemoveHs``) if you want heavy-atom RMSD, as this function
             operates on all atoms present in the molecule.
        prealigned: If True, skip Kabsch alignment and compute RMSD on raw
                    coordinates.  If False (default), optimally align each pair.
        stream: CUDA stream to use.  If None, uses the current stream.

    Returns:
        AsyncGpuResult wrapping a 1-D tensor of shape ``(N*(N-1)/2,)`` containing
        RMSD values in lower-triangle condensed order.  The RMSD for conformer
        pair (i, j) with i > j is at index ``i*(i-1)//2 + j``.

    Raises:
        ValueError: If ``mol`` is None.
        TypeError: If ``stream`` is not a ``torch.cuda.Stream`` or None.

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import rdDistGeom
        >>> from nvmolkit.conformerRmsd import GetConformerRMSMatrix
        >>> from nvmolkit.clustering import butina
        >>>
        >>> mol = Chem.AddHs(Chem.MolFromSmiles('CCCCCC'))
        >>> rdDistGeom.EmbedMultipleConfs(mol, numConfs=50)
        >>> no_h = Chem.RemoveHs(mol)
        >>>
        >>> # Compute RMSD matrix on GPU
        >>> rmsd_matrix = GetConformerRMSMatrix(no_h)
        >>>
        >>> # Reshape to square for GPU Butina clustering
        >>> import torch
        >>> torch.cuda.synchronize()
        >>> n = no_h.GetNumConformers()
        >>> square = torch.zeros(n, n, device='cuda', dtype=torch.float64)
        >>> idx = torch.tril_indices(n, n, offset=-1)
        >>> square[idx[0], idx[1]] = rmsd_matrix.torch()
        >>> square = square + square.T
        >>> clusters = butina(square, cutoff=0.5)
    """
    if mol is None:
        raise ValueError("mol must not be None")
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(f"stream must be a torch.cuda.Stream or None, got {type(stream).__name__}")

    stream_ptr = (stream if stream is not None else torch.cuda.current_stream()).cuda_stream
    return AsyncGpuResult(_conformerRmsd.GetConformerRMSMatrix(mol, prealigned, stream_ptr))


def GetConformerRMSMatrixBatch(
    mols: list["Mol"],
    prealigned: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> list[AsyncGpuResult]:
    """Compute pairwise RMSD matrices for a batch of molecules on GPU.

    All molecules are processed in a single kernel launch, so their conformer
    pairs execute concurrently.  This improves GPU saturation over repeated
    single-molecule calls, particularly for molecules with few conformers.

    Args:
        mols: List of RDKit molecules, each with zero or more conformers.
              Strip hydrogens first (``Chem.RemoveHs``) for heavy-atom RMSD.
              Molecules with fewer than 2 conformers return an empty result.
        prealigned: If True, skip Kabsch alignment and compute RMSD on raw
                    coordinates.  If False (default), optimally align each pair.
        stream: CUDA stream to use.  If None, uses the current stream.

    Returns:
        List of :class:`AsyncGpuResult`, one per input molecule, in the same
        order as ``mols``.  Each element wraps a 1-D tensor of shape
        ``(N*(N-1)/2,)`` for a molecule with N conformers, or shape ``(0,)``
        for molecules with fewer than 2 conformers.  The RMSD for conformer
        pair (i, j) with i > j is at index ``i*(i-1)//2 + j``.

    Raises:
        ValueError: If any element of ``mols`` is None.
        TypeError: If ``stream`` is not a ``torch.cuda.Stream`` or None.

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import rdDistGeom
        >>> from nvmolkit.conformerRmsd import GetConformerRMSMatrixBatch
        >>>
        >>> mols = [Chem.RemoveHs(Chem.AddHs(Chem.MolFromSmiles(s)))
        ...         for s in ["CCCCCC", "c1ccccc1"]]
        >>> for mol in mols:
        ...     rdDistGeom.EmbedMultipleConfs(mol, numConfs=20)
        >>>
        >>> results = GetConformerRMSMatrixBatch(mols)
        >>> # results[0] is AsyncGpuResult for mols[0], etc.
    """
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(f"stream must be a torch.cuda.Stream or None, got {type(stream).__name__}")

    for i, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"mol at index {i} must not be None")

    stream_ptr = (stream if stream is not None else torch.cuda.current_stream()).cuda_stream
    raw = _conformerRmsd.GetConformerRMSMatrixBatch(mols, prealigned, stream_ptr)
    return [AsyncGpuResult(r) for r in raw]
