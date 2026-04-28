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

"""DEVICE-mode tests for the batched forcefield wrappers (issue #140).

These tests exercise the on-device output path: ``compute_energy(output=DEVICE)``,
``compute_gradients(output=DEVICE)``, ``minimize(output=DEVICE)``, and the
``positions()`` / ``indices()`` accessors on both
:class:`MMFFBatchedForcefield` and :class:`UFFBatchedForcefield`.

Each test compares the DEVICE-mode result against the host-mode result on
identical input to confirm the device path returns the same numbers.
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from nvmolkit.batchedForcefield import MMFFBatchedForcefield, UFFBatchedForcefield
from nvmolkit.types import CoordinateOutput


def make_embedded_mol(smiles: str, num_confs: int = 1, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def _flatten(nested):
    return [v for inner in nested for v in inner]


@pytest.fixture(scope="module")
def two_mols():
    return [
        make_embedded_mol("CCO", num_confs=2, seed=1),
        make_embedded_mol("CCCC", num_confs=3, seed=2),
    ]


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_compute_energy_device_matches_host(ff_cls, two_mols):
    ff = ff_cls(two_mols)
    host_nested = ff.compute_energy()
    host_flat = _flatten(host_nested)

    device_handle = ff.compute_energy(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    device_flat = device_handle.numpy().tolist()

    assert len(device_flat) == len(host_flat)
    for h, d in zip(host_flat, device_flat):
        assert abs(h - d) < 1e-9


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_compute_gradients_device_matches_host(ff_cls, two_mols):
    ff = ff_cls(two_mols)
    host_nested = ff.compute_gradients()
    # host_nested[mol][conf] is a flat [x0,y0,z0, ...] vector
    host_flat = []
    for per_mol in host_nested:
        for per_conf in per_mol:
            host_flat.extend(per_conf)

    device_handle = ff.compute_gradients(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    device_flat = device_handle.numpy().tolist()

    assert len(device_flat) == len(host_flat)
    for h, d in zip(host_flat, device_flat):
        assert abs(h - d) < 1e-9


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_indices_match_layout(ff_cls, two_mols):
    ff = ff_cls(two_mols)
    ff._ensure_built()  # force build
    atom_starts, mol_indices, conf_indices = ff.indices()
    torch.cuda.synchronize()
    atom_starts_np = atom_starts.numpy()
    mol_indices_np = mol_indices.numpy()
    conf_indices_np = conf_indices.numpy()

    expected_n_conformers = sum(m.GetNumConformers() for m in two_mols)
    assert len(mol_indices_np) == expected_n_conformers
    assert len(conf_indices_np) == expected_n_conformers
    assert len(atom_starts_np) == expected_n_conformers + 1
    assert atom_starts_np[0] == 0

    cursor = 0
    idx = 0
    for mol_idx, mol in enumerate(two_mols):
        natoms = mol.GetNumAtoms()
        for conf_idx in range(mol.GetNumConformers()):
            assert mol_indices_np[idx] == mol_idx
            assert conf_indices_np[idx] == conf_idx
            cursor += natoms
            assert atom_starts_np[idx + 1] == cursor
            idx += 1


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_positions_accessor_reflects_minimize(ff_cls, two_mols):
    ff = ff_cls(two_mols)
    pos_before = ff.positions().numpy().copy()

    # Run device minimize - should refresh the wrapper's persistent positions
    # buffer in place without touching the RDKit conformers.
    result = ff.minimize(maxIters=15, output=CoordinateOutput.DEVICE)
    pos_after = ff.positions().numpy()

    assert pos_before.shape == pos_after.shape
    assert pos_after.shape == result.positions.numpy().shape
    # After a real minimize step the positions should change for at least one
    # coord; assert the persistent buffer is in sync with the device result.
    assert (pos_after == result.positions.numpy()).all()


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_minimize_device_energies_match_host(ff_cls, two_mols):
    # Run host-mode minimize on a separate copy; energies should match the
    # device-mode minimize energies within BFGS tolerance.
    mols_host = [Chem.Mol(m) for m in two_mols]
    mols_device = [Chem.Mol(m) for m in two_mols]

    ff_host = ff_cls(mols_host)
    energies_host_nested, _ = ff_host.minimize(maxIters=20)
    energies_host = _flatten(energies_host_nested)

    ff_device = ff_cls(mols_device)
    result = ff_device.minimize(maxIters=20, output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    energies_device = result.energies.numpy().tolist()

    assert len(energies_device) == len(energies_host)
    for h, d in zip(energies_host, energies_device):
        assert abs(h - d) < 1e-3, f"host {h} vs device {d}"
