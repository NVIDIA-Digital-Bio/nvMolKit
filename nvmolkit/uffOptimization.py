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
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""GPU-accelerated UFF optimization for molecular conformers."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rdkit.Chem import rdForceFieldHelpers

if TYPE_CHECKING:
    from rdkit.Chem import Mol

from nvmolkit import _uffOptimization
from nvmolkit.types import HardwareOptions


def UFFOptimizeMoleculesConfs(
    molecules: list["Mol"],
    maxIters: int = 1000,
    vdwThreshold: float | Sequence[float] = 10.0,
    ignoreInterfragInteractions: bool | Sequence[bool] = True,
    hardwareOptions: HardwareOptions | None = None,
) -> list[list[float]]:
    """Optimize conformers for multiple molecules using the UFF force field.

    Args:
        molecules: List of RDKit molecules to optimize. Each molecule should already
            contain conformers.
        maxIters: Maximum number of UFF minimization iterations.
        vdwThreshold: Van der Waals threshold used when constructing the UFF force
            field. May be provided as a scalar or per-molecule sequence.
        ignoreInterfragInteractions: If ``True``, omit non-bonded terms between
            fragments. May be provided as a scalar or per-molecule sequence.
        hardwareOptions: Configures CPU and GPU batching, threading, and device
            selection. Defaults are chosen automatically when omitted.

    Returns:
        List of lists of energies, where each inner list contains optimized
        conformer energies for the corresponding molecule.

    Raises:
        ValueError: If molecules contains ``None`` entries or molecules lacking UFF
            atom types. ``e.args[1]`` contains keys ``"none"`` and ``"no_params"``.
    """
    if not molecules:
        return []

    none_indices = []
    no_params_indices = []
    for i, mol in enumerate(molecules):
        if mol is None:
            none_indices.append(i)
        elif not rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            no_params_indices.append(i)

    if none_indices or no_params_indices:
        parts = []
        if none_indices:
            parts.append(f"None at indices {none_indices}")
        if no_params_indices:
            parts.append(f"lacking UFF atom types at indices {no_params_indices}")
        raise ValueError(
            "; ".join(parts),
            {"none": none_indices, "no_params": no_params_indices},
        )

    def _normalize_scalar_or_list(value, name: str):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != len(molecules):
                raise ValueError(f"Expected {len(molecules)} values for {name}, got {len(value)}")
            return list(value)
        return [value for _ in molecules]

    thresholds = [float(v) for v in _normalize_scalar_or_list(vdwThreshold, "vdwThreshold")]
    interfrag_flags = [
        bool(v)
        for v in _normalize_scalar_or_list(
            ignoreInterfragInteractions, "ignoreInterfragInteractions"
        )
    ]

    if hardwareOptions is None:
        hardwareOptions = HardwareOptions()
    return _uffOptimization.UFFOptimizeMoleculesConfs(
        molecules,
        int(maxIters),
        thresholds,
        interfrag_flags,
        hardwareOptions._as_native(),
    )
