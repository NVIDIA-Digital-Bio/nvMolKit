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

"""Types facilitating GPU-accelerated operations."""

from enum import Enum
from typing import Iterable, List, Optional

import torch

from nvmolkit import _embedMolecules  # type: ignore


class HardwareOptions:
    """Configures GPU hardware settings for batch operations.

    Use this class to control threading and batching behavior for GPU-accelerated
    workflows such as ETKDG embedding and MMFF optimization.

    Parameters:
        preprocessingThreads: Number of CPU threads for preprocessing. Use -1 to auto-detect and use all CPU threads.
        batchSize: Number of conformers processed per batch. -1 for default.
        batchesPerGpu: Number of batches processed concurrently on each GPU. Use -1 for default.
        gpuIds: GPU device IDs to target. Provide an empty list to use all available GPUs.
    """

    def __init__(
        self,
        preprocessingThreads: int = -1,
        batchSize: int = -1,
        batchesPerGpu: int = -1,
        gpuIds: Iterable[int] | None = None,
    ) -> None:
        if _embedMolecules is None:  # propagate real import failure early
            raise ImportError("nvmolkit._embedMolecules is not available; build native extensions")
        native = _embedMolecules.BatchHardwareOptions()
        native.preprocessingThreads = int(preprocessingThreads)
        native.batchSize = int(batchSize)
        native.gpuIds = list(gpuIds) if gpuIds is not None else []
        self._native = native
        self.batchesPerGpu = batchesPerGpu  # reuses setter validation

    @property
    def preprocessingThreads(self) -> int:
        """Number of CPU threads for preprocessing. -1 auto-detects."""
        return self._native.preprocessingThreads

    @preprocessingThreads.setter
    def preprocessingThreads(self, value: int) -> None:
        self._native.preprocessingThreads = int(value)

    @property
    def batchSize(self) -> int:
        """Number of conformers per batch. -1 selects an auto-tuned value."""
        return self._native.batchSize

    @batchSize.setter
    def batchSize(self, value: int) -> None:
        self._native.batchSize = int(value)

    @property
    def batchesPerGpu(self) -> int:
        """Batches processed concurrently on each GPU. -1 auto."""
        return self._native.batchesPerGpu

    @batchesPerGpu.setter
    def batchesPerGpu(self, value: int) -> None:
        value = int(value)
        if value != -1 and value <= 0:
            raise ValueError("batchesPerGpu must be greater than 0 or -1 for automatic")
        self._native.batchesPerGpu = value

    @property
    def gpuIds(self) -> List[int]:
        """GPU device IDs to target. Empty list uses all available GPUs."""
        return list(self._native.gpuIds)

    @gpuIds.setter
    def gpuIds(self, value: Iterable[int]) -> None:
        self._native.gpuIds = list(value)

    def _as_native(self):
        """Internal: return the underlying BatchHardwareOptions object."""
        return self._native


class AsyncGpuResult:
    """Handle to a GPU result.

    Populates the ``__cuda_array_interface__`` attribute which can be consumed by other libraries. Note that
    this result is async, and the data cannot be accessed without a sync, such as ``torch.cuda.synchronize()``.
    """

    def __init__(self, obj, gpu_id: Optional[int] = None):
        """Internal construction of the AsyncGpuResult object.

        Args:
            obj: An object exposing ``__cuda_array_interface__``.
            gpu_id: Optional GPU device id where the underlying buffer lives. If omitted, torch
                infers the device from the CUDA pointer attributes (typically the current device).
        """
        if not hasattr(obj, "__cuda_array_interface__"):
            raise TypeError(f"Object {obj} does not have a __cuda_array_interface__ attribute")
        device = "cuda" if gpu_id is None else f"cuda:{int(gpu_id)}"
        self.arr = torch.as_tensor(obj, device=device)

    @property
    def __cuda_array_interface__(self):
        """Return the CUDA array interface for the underlying data."""
        return self.arr.__cuda_array_interface__

    @property
    def device(self):
        """Return the device of the underlying data."""
        return self.arr.device

    def torch(self):
        """Return the underlying data as a torch tensor. This is an asynchronous operation."""
        return self.arr

    def numpy(self):
        """Return the underlying data as a numpy array. This is a blocking operation."""
        torch.cuda.synchronize()
        return self.arr.cpu().numpy()


class CoordinateOutput(Enum):
    """Selects how conformer-producing APIs return optimized coordinates.

    - ``RDKIT_CONFORMERS``: legacy behavior. Optimized coordinates are written back into each
      input molecule's RDKit conformer list and energies (where applicable) are returned as
      Python lists.
    - ``DEVICE``: coordinates and (where applicable) energies stay on the GPU and are returned
      as a :class:`DeviceCoordResult`. Use this to chain GPU-accelerated workflows (e.g. ETKDG
      followed by MMFF) without host round-trips.
    """

    RDKIT_CONFORMERS = "rdkit"
    DEVICE = "device"


class DeviceCoordResult:
    """On-device, flat CSR-style result of a conformer-producing GPU pipeline.

    All buffers live on a single GPU identified by :attr:`gpu_id`. Sizes are linked as follows:

    - :attr:`positions` has shape ``(total_atoms, 3)`` (float64).
    - :attr:`atom_starts` has shape ``(n_conformers + 1,)`` (int32). The slice
      ``positions[atom_starts[i]:atom_starts[i+1]]`` holds conformer ``i``'s atoms.
    - :attr:`mol_indices` has shape ``(n_conformers,)`` (int32) and maps each conformer to its
      input molecule index.
    - :attr:`conf_indices` has shape ``(n_conformers,)`` (int32) and gives each conformer's
      per-molecule conformer index.
    - :attr:`energies` (MMFF/UFF only) has shape ``(n_conformers,)`` (float64).
    - :attr:`converged` (MMFF/UFF only) has shape ``(n_conformers,)`` (int8; 1 = converged).

    Buffers are exposed as :class:`AsyncGpuResult` to enable zero-copy interoperability with
    PyTorch / CuPy. Synchronize before consuming on the host (e.g. ``torch.cuda.synchronize()``).
    """

    def __init__(
        self,
        positions: AsyncGpuResult,
        atom_starts: AsyncGpuResult,
        mol_indices: AsyncGpuResult,
        conf_indices: AsyncGpuResult,
        gpu_id: int,
        energies: Optional[AsyncGpuResult] = None,
        converged: Optional[AsyncGpuResult] = None,
    ) -> None:
        self.positions = positions
        self.atom_starts = atom_starts
        self.mol_indices = mol_indices
        self.conf_indices = conf_indices
        self.energies = energies
        self.converged = converged
        self.gpu_id = int(gpu_id)

    @property
    def num_conformers(self) -> int:
        """Total number of conformers represented in this result."""
        return int(self.atom_starts.torch().numel()) - 1

    def per_molecule(self) -> List[List["torch.Tensor"]]:
        """Return a nested list of per-molecule, per-conformer position views.

        The outer list is indexed by input molecule index; the inner list contains one
        ``(n_atoms, 3)`` torch view per conformer for that molecule. Views share storage with
        :attr:`positions` (no copy). Reading the index tensors via ``.tolist()`` implicitly
        synchronizes; reading position values still requires the caller to synchronize as needed.
        """
        positions = self.positions.torch()
        atom_starts = self.atom_starts.torch().tolist()
        mol_indices = self.mol_indices.torch().tolist()
        if not mol_indices:
            return []
        max_mol = max(mol_indices)
        result: List[List[torch.Tensor]] = [[] for _ in range(max_mol + 1)]
        for conf_idx, mol_idx in enumerate(mol_indices):
            start = atom_starts[conf_idx]
            stop = atom_starts[conf_idx + 1]
            result[mol_idx].append(positions[start:stop])
        return result

    def dense(self, pad_value: float = float("nan")) -> "torch.Tensor":
        """Materialize a padded dense ``(n_conformers, max_atoms, 3)`` tensor (float64).

        Conformers smaller than ``max_atoms`` are padded with ``pad_value`` along the atom axis.
        This always allocates a fresh tensor on the same GPU as :attr:`positions`. The size
        computation reads the small index tensors back to host and therefore synchronizes.
        """
        positions = self.positions.torch()
        atom_starts = self.atom_starts.torch()
        n_conformers = atom_starts.numel() - 1
        if n_conformers == 0:
            return torch.empty((0, 0, 3), dtype=positions.dtype, device=positions.device)
        sizes = (atom_starts[1:] - atom_starts[:-1]).tolist()
        max_atoms = max(sizes) if sizes else 0
        out = torch.full(
            (n_conformers, max_atoms, 3),
            pad_value,
            dtype=positions.dtype,
            device=positions.device,
        )
        starts = atom_starts.tolist()
        for conf_idx, conf_size in enumerate(sizes):
            start = starts[conf_idx]
            out[conf_idx, :conf_size, :] = positions[start : start + conf_size]
        return out
