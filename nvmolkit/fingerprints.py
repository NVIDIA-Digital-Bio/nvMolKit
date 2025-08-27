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

"""GPU-accelerated fingerprint generation."""
import torch

from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit._Fingerprints import MorganFingerprintGenerator as InternalFPGen
from nvmolkit.types import AsyncGpuResult

def unpack_fingerprint(fp: torch.Tensor) -> torch.Tensor:
    """Unpack an integer-encoded fingerprint into a 2D boolean tensor of shape (len(fp), fingerprint_size).

    Args:
        fp: A tensor of shape `(n_fps, fp_size / 32)`, containing packed fingerprints with dtype int32

    Returns:
        A boolean tensor of shape `(n_fps, fp_size)`
    """
    n_fps = fp.shape[0]
    n_ints = fp.shape[1]
    fp_size = n_ints * 32
    return ((fp.unsqueeze(2) >> torch.arange(0, 32, device=fp.device)) & 1).bool().reshape(n_fps, fp_size)


def pack_fingerprint(fp: torch.Tensor) -> torch.Tensor:
    """Pack a 2D boolean tensor of shape `(n_fps, fingerprint_size)` into an integer-encoded fingerprint.

    Args:
        fp: A boolean tensor of shape `(n_fps, fp_size)`

    Returns:
        A tensor of shape `(n_fps, fp_size / 32)` containing packed fingerprints (rounded up to the nearest multiple of 32)
    """
    n_fps, fp_size = fp.shape
    n_ints = (fp_size + 31) // 32  # Number of int32s needed per fingerprint

    # Pad to next multiple of 32 if needed
    if fp_size % 32 != 0:
        padded_size = n_ints * 32
        padded = torch.zeros((n_fps, padded_size), dtype=torch.bool, device=fp.device)
        padded[:, :fp_size] = fp
        fp = padded

    # Reshape to group bits into 32-bit chunks
    fp_reshaped = fp.reshape(n_fps, n_ints, 32)

    # Create powers of 2 for each bit position, using 0 to 31 instead of 31 to 0 to fix endianness
    powers = 1 << torch.arange(0, 32, device=fp.device)

    # Multiply and sum to create packed integers
    return (fp_reshaped * powers.unsqueeze(0)).sum(dim=2)

class MorganFingerprintGenerator:
    """Morgan fingerprint generator."""
    def __init__(self, radius: int, fpSize: int):
        """Initialize the Morgan fingerprint generator.

        Args:
            radius: The radius of the Morgan fingerprint.
            fpSize: The size of the fingerprint. Must be one of {128, 256, 512, 1024, 2048}.
        """
        self._internal = InternalFPGen(radius, fpSize)

    def GetFingerprints(self, mols: list, num_threads: int = 0):
        """Compute Morgan fingerprints for a list of molecules.

        Preprocessing of fingerprinting features is done on the CPU, and is parallelized with the `num_threads` argument.
        The resulting tensor has dtype torch.int32 and contains a packed fingerprint for each molecule, one row per molecule.

        Packed fingerprints can be passed directly to nvMolKit similarity calculations, or unpacked
        via `unpack_fingerprint`.

        Args:
            mols: List of RDKit molecules to generate fingerprints for
            num_threads: Number of CPU threads to use for fingerprint generation. If 0, uses all available threads.

        Returns:
            AsyncGpuResult wrapping a torch.Tensor of shape (len(mols), fpSize / 32) containing the fingerprints.
            Each row is a fingerprint for the corresponding molecule.
        """
        return AsyncGpuResult(self._internal.GetFingerprintsDevice(mols, num_threads))
