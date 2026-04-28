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

"""End-to-end DEVICE-mode tests for the top-level Python entry points (issue #140).

Covers:
- ``EmbedMolecules(output=DEVICE)`` returns a populated DeviceCoordResult.
- ``MMFFOptimizeMoleculesConfs(output=DEVICE)`` and ``UFFOptimizeMoleculesConfs(output=DEVICE)``
  return DeviceCoordResults whose energies match the host-mode energies.
- ETKDG -> MMFF chaining via device_input keeps coordinates on the GPU
  end-to-end and produces correct shapes.
- ``DeviceCoordResult.per_molecule()`` and ``.dense()`` view helpers work on
  results returned from the entry points.
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from nvmolkit.embedMolecules import EmbedMolecules
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
from nvmolkit.types import CoordinateOutput, DeviceCoordResult, HardwareOptions
from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs


def _embed(smiles: str, n_confs: int = 1, seed: int = 1234):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = seed
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    return mol


def _single_thread_options():
    return HardwareOptions(preprocessingThreads=1, batchSize=64, batchesPerGpu=1, gpuIds=[0])


def test_embed_molecules_device_returns_device_coord_result():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    params.pruneRmsThresh = -1.0

    result = EmbedMolecules(
        [mol],
        params,
        confsPerMolecule=1,
        hardwareOptions=_single_thread_options(),
        output=CoordinateOutput.DEVICE,
    )
    assert isinstance(result, DeviceCoordResult)
    assert result.gpu_id == 0
    assert mol.GetNumConformers() == 0  # device mode does not modify the host mol

    torch.cuda.synchronize()
    positions = result.positions.numpy()
    assert positions.shape == (mol.GetNumAtoms(), 3)
    assert torch.isfinite(torch.from_numpy(positions)).all()


def test_embed_molecules_rdkit_returns_none():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42

    result = EmbedMolecules(
        [mol],
        params,
        confsPerMolecule=1,
        hardwareOptions=_single_thread_options(),
        output=CoordinateOutput.RDKIT_CONFORMERS,
    )
    assert result is None
    assert mol.GetNumConformers() == 1


def test_mmff_device_energies_match_host():
    mol_host = _embed("CCO", n_confs=2, seed=11)
    mol_dev = _embed("CCO", n_confs=2, seed=11)
    assert mol_host.GetNumConformers() == mol_dev.GetNumConformers() == 2

    energies_host = MMFFOptimizeMoleculesConfs(
        [mol_host], maxIters=30, hardwareOptions=_single_thread_options()
    )
    flat_host = [e for inner in energies_host for e in inner]

    result = MMFFOptimizeMoleculesConfs(
        [mol_dev],
        maxIters=30,
        hardwareOptions=_single_thread_options(),
        output=CoordinateOutput.DEVICE,
    )
    assert isinstance(result, DeviceCoordResult)
    torch.cuda.synchronize()
    flat_dev = result.energies.numpy().tolist()
    assert len(flat_dev) == len(flat_host)
    for h, d in zip(flat_host, flat_dev):
        assert abs(h - d) < 1e-3, f"host {h} vs device {d}"


def test_uff_device_energies_match_host():
    mol_host = _embed("CCO", n_confs=2, seed=22)
    mol_dev = _embed("CCO", n_confs=2, seed=22)

    energies_host = UFFOptimizeMoleculesConfs(
        [mol_host], maxIters=50, hardwareOptions=_single_thread_options()
    )
    flat_host = [e for inner in energies_host for e in inner]

    result = UFFOptimizeMoleculesConfs(
        [mol_dev],
        maxIters=50,
        hardwareOptions=_single_thread_options(),
        output=CoordinateOutput.DEVICE,
    )
    assert isinstance(result, DeviceCoordResult)
    torch.cuda.synchronize()
    flat_dev = result.energies.numpy().tolist()
    assert len(flat_dev) == len(flat_host)
    for h, d in zip(flat_host, flat_dev):
        assert abs(h - d) < 1e-3, f"host {h} vs device {d}"


def test_etkdg_to_mmff_chained_device_input():
    """Chain ETKDG-DEVICE -> MMFF-DEVICE-INPUT-DEVICE-OUTPUT.

    The MMFF call needs at least one RDKit conformer per mol for force-field construction
    (used only for hybridization/VDW special-case checks); the actual starting positions come
    from the ETKDG device result.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    # Embed on host first - we need a conformer to seed FF construction. The position values
    # don't matter for the chained device-input flow; only the existence of one conformer.
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 99
    rdDistGeom.EmbedMolecule(mol, params=params)
    assert mol.GetNumConformers() == 1

    # Run ETKDG with DEVICE output to get fresh on-device starting coordinates.
    etkdg_params = rdDistGeom.ETKDGv3()
    etkdg_params.useRandomCoords = True
    etkdg_params.randomSeed = 99
    etkdg_params.pruneRmsThresh = -1.0
    etkdg_result = EmbedMolecules(
        [mol],
        etkdg_params,
        confsPerMolecule=1,
        hardwareOptions=_single_thread_options(),
        output=CoordinateOutput.DEVICE,
    )
    assert isinstance(etkdg_result, DeviceCoordResult)
    assert etkdg_result.num_conformers == 1

    # Feed it back to MMFF (also DEVICE). The mol still has its 1 host conformer for FF
    # construction; that is the documented requirement.
    mmff_result = MMFFOptimizeMoleculesConfs(
        [mol],
        maxIters=30,
        hardwareOptions=_single_thread_options(),
        output=CoordinateOutput.DEVICE,
        device_input=etkdg_result,
    )
    assert isinstance(mmff_result, DeviceCoordResult)
    torch.cuda.synchronize()
    assert mmff_result.energies.numpy().shape == (1,)
    pos = mmff_result.positions.numpy()
    assert pos.shape == (mol.GetNumAtoms(), 3)


def test_device_input_requires_native_handle():
    mol = _embed("CCO", n_confs=1, seed=7)

    # Build a DeviceCoordResult directly from torch tensors - it lacks _native_handle.
    positions = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float64, device="cuda")
    atom_starts = torch.tensor([0, mol.GetNumAtoms()], dtype=torch.int32, device="cuda")
    mol_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    conf_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    from nvmolkit.types import AsyncGpuResult
    fake = DeviceCoordResult(
        positions=AsyncGpuResult(positions),
        atom_starts=AsyncGpuResult(atom_starts),
        mol_indices=AsyncGpuResult(mol_indices),
        conf_indices=AsyncGpuResult(conf_indices),
        gpu_id=0,
    )
    with pytest.raises(Exception, match="native handle|nvmolkit.types.DeviceCoordResult"):
        MMFFOptimizeMoleculesConfs(
            [mol],
            maxIters=5,
            hardwareOptions=_single_thread_options(),
            output=CoordinateOutput.DEVICE,
            device_input=fake,
        )


def test_per_molecule_and_dense_views():
    mols = [_embed("CCO", n_confs=2, seed=3), _embed("CCC", n_confs=1, seed=4)]
    result = MMFFOptimizeMoleculesConfs(
        mols,
        maxIters=10,
        hardwareOptions=_single_thread_options(),
        output=CoordinateOutput.DEVICE,
    )
    assert isinstance(result, DeviceCoordResult)
    torch.cuda.synchronize()

    per_mol = result.per_molecule()
    assert len(per_mol) == 2
    assert len(per_mol[0]) == 2
    assert len(per_mol[1]) == 1
    assert per_mol[0][0].shape == (mols[0].GetNumAtoms(), 3)
    assert per_mol[1][0].shape == (mols[1].GetNumAtoms(), 3)

    dense = result.dense()
    assert dense.shape[0] == 3
    assert dense.shape[2] == 3
    assert dense.shape[1] == max(mols[0].GetNumAtoms(), mols[1].GetNumAtoms())


def test_device_input_only_with_device_output():
    mol = _embed("CCO", n_confs=1, seed=9)
    # Construct a fake handle just to trigger the validation - the function should reject it
    # before ever looking at the handle.
    with pytest.raises(ValueError, match="device_input is only supported"):
        MMFFOptimizeMoleculesConfs(
            [mol],
            maxIters=5,
            hardwareOptions=_single_thread_options(),
            output=CoordinateOutput.RDKIT_CONFORMERS,
            device_input=object(),  # type: ignore[arg-type]
        )
