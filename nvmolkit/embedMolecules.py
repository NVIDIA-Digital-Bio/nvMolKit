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

"""GPU-accelerated ETKDG conformer generation for multiple molecules.

This module provides GPU-accelerated implementations of ETKDG (Experimental-Torsion-Knowledge Distance-Geometry) conformer generation for multiple molecules using CUDA and OpenMP.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.Chem.rdDistGeom import EmbedParameters

__all__ = ["EmbedMolecules"]

from nvmolkit import _embedMolecules  # type: ignore
from nvmolkit.types import HardwareOptions


def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[HardwareOptions] = None,
) -> None:
    """Embed multiple molecules with multiple conformers on GPUs.

    This function performs GPU-accelerated ETKDG conformer generation on multiple molecules.
    It uses CUDA for GPU acceleration and OpenMP for CPU parallelization to achieve high
    performance embedding of large molecule sets.

    Args:
        molecules: List of RDKit molecules to embed. Molecules should be prepared
                  (sanitized, explicit hydrogens added if needed).
        params: RDKit EmbedParameters object with embedding settings. Must have
               useRandomCoords=True for ETKDG.
        confsPerMolecule: Number of conformers to generate per molecule (default: 1)
        maxIterations: Maximum ETKDG iterations, -1 for automatic calculation (default: -1)
        hardwareOptions: HardwareOptions with hardware settings. If None, uses defaults.

    Returns:
        None. Input molecules are modified in-place with generated conformers.

    Raises:
        ValueError: If any molecule in the input list is invalid, or if hardware
                   configuration parameters are invalid
        RuntimeError: If CUDA operations fail or embedding encounters errors

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem.rdDistGeom import ETKDGv3
        >>> from nvmolkit.types import HardwareOptions
        >>> import nvmolkit.embedMolecules as embed
        >>>
        >>> # Load molecules
        >>> mol1 = Chem.AddHs(Chem.MolFromSmiles('CCO'))
        >>> mol2 = Chem.AddHs(Chem.MolFromSmiles('CCC'))
        >>>
        >>> # Set up embedding parameters
        >>> params = ETKDGv3()
        >>> params.useRandomCoords = True  # Required for nvMolKit ETKDG
        >>>
        >>> # Configure hardware options
        >>> hardware_opts = HardwareOptions(
        ...     preprocessingThreads=8,
        ...     batchSize=500,
        ...     batchesPerGpu=4,
        ...     gpuIds=[0, 1],
        ... )
        >>> embed.EmbedMolecules([mol1, mol2], params, confsPerMolecule=5, hardwareOptions=hardware_opts)
        >>>
        >>> # Check conformers were generated
        >>> mol1.GetNumConformers()  # Should be 5
        >>> mol2.GetNumConformers()  # Should be 5

    Note:
        - Input molecules are modified in-place with generated conformers
        - params.useRandomCoords must be True for ETKDG algorithm
        - If gpuIds is empty, all available GPUs (0 to N-1) will be used automatically
    """
    # Validate input
    if not molecules:
        return

    for i, mol in enumerate(molecules):
        if mol is None:
            raise ValueError(f"Molecule at index {i} is None")

    if not params.useRandomCoords:
        raise ValueError("ETKDG requires useRandomCoords=True in EmbedParameters")

    # Use default hardware options if none provided
    if hardwareOptions is None:
        hardwareOptions = HardwareOptions()
    native_options = hardwareOptions._as_native()

    # Validate hardware options

    if hardwareOptions.batchesPerGpu <= 0 and hardwareOptions.batchesPerGpu != -1:
        raise ValueError("batchesPerGpu must be greater than 0 or -1 for automatic")

    # Call the C++ implementation
    _embedMolecules.EmbedMolecules(molecules, params, confsPerMolecule, maxIterations, native_options)
