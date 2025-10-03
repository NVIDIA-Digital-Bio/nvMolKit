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
import enum
import torch
from typing import Iterable, List
from nvmolkit import _embedMolecules  # type: ignore
from nvmolkit import _mmffOptimization  # type: ignore


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
        native.batchesPerGpu = int(batchesPerGpu)
        native.gpuIds = list(gpuIds) if gpuIds is not None else []
        self._native = native

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
        self._native.batchesPerGpu = int(value)

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


class OptimizerBackend(enum.Enum):
    """Enumeration of supported MMFF optimizer backends."""

    BFGS = _mmffOptimization.OptimizerBackend.BFGS
    FIRE = _mmffOptimization.OptimizerBackend.FIRE


class OptimizerOptions:
    """Configures the MMFF minimizer backend.

    Parameters:
        backend: Choice of numerical optimizer. Use ``OptimizerBackend.BFGS`` for
            BFGS (default) or ``OptimizerBackend.FIRE`` for the
            Fast Inertial Relaxation Engine.
    """

    def __init__(self, backend: OptimizerBackend | None = None) -> None:
        self._native = _mmffOptimization.OptimizerOptions()
        if backend is not None:
            self._native.backend = backend.value

    @property
    def backend(self) -> OptimizerBackend:
        """Selected optimizer backend."""
        return OptimizerBackend(self._native.backend)

    @backend.setter
    def backend(self, value: OptimizerBackend | int) -> None:
        backend = OptimizerBackend(value) if not isinstance(value, OptimizerBackend) else value
        self._native.backend = backend.value


    def _as_native(self):
        """Internal: return the underlying OptimizerOptions object."""
        return self._native


class AsyncGpuResult:
    """Handle to a GPU result.

    Populates the __cuda_array_interface__ attribute which can be consumed by other libraries. Note that
    this result is async, and the data cannot be accessed without a sync, such as torch.cuda.synchronize().

    # TODO: Handle devices and streams.
    """
    def __init__(self, obj):
        """Internal construction of the AsyncGpuResult object."""
        if not hasattr(obj, '__cuda_array_interface__'):
            raise TypeError(f"Object {obj} does not have a __cuda_array_interface__ attribute")
        self.arr = torch.as_tensor(obj, device='cuda')

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
