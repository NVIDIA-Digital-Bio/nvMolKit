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
``compute_gradients(output=DEVICE)``, and ``minimize(output=DEVICE)`` on both
:class:`MMFFBatchedForcefield` and :class:`UFFBatchedForcefield`.

Each test compares the DEVICE-mode result against the host-mode result on
identical input to confirm the device path returns the same numbers.
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from nvmolkit.batchedForcefield import MMFFBatchedForcefield, UFFBatchedForcefield
from nvmolkit.types import CoordinateOutput, Device3DResult, DevicePerConfResult


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

    device_result = ff.compute_energy(output=CoordinateOutput.DEVICE)
    assert isinstance(device_result, DevicePerConfResult)
    torch.cuda.synchronize()
    device_nested = device_result.per_molecule()

    assert len(device_nested) == len(host_nested)
    for h_inner, d_inner in zip(host_nested, device_nested):
        assert len(h_inner) == len(d_inner)
        for h, d in zip(h_inner, d_inner):
            assert abs(h - d) < 1e-9


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_compute_energy_device_indices_match_layout(ff_cls, two_mols):
    """DevicePerConfResult.mol_indices/conf_indices match the host conformer flattening."""
    ff = ff_cls(two_mols)
    device_result = ff.compute_energy(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    mol_indices = device_result.mol_indices.torch().tolist()
    conf_indices = device_result.conf_indices.torch().tolist()

    expected_mol_indices = []
    expected_conf_indices = []
    for mol_idx, mol in enumerate(two_mols):
        for conf_idx in range(mol.GetNumConformers()):
            expected_mol_indices.append(mol_idx)
            expected_conf_indices.append(conf_idx)
    assert mol_indices == expected_mol_indices
    assert conf_indices == expected_conf_indices
    assert device_result.n_mols == len(two_mols)


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_compute_gradients_device_matches_host(ff_cls, two_mols):
    ff = ff_cls(two_mols)
    host_nested = ff.compute_gradients()
    host_flat = []
    for per_mol in host_nested:
        for per_conf in per_mol:
            host_flat.extend(per_conf)

    device_result = ff.compute_gradients(output=CoordinateOutput.DEVICE)
    assert isinstance(device_result, Device3DResult)
    torch.cuda.synchronize()
    device_flat = device_result.values.torch().reshape(-1).tolist()

    assert len(device_flat) == len(host_flat)
    for h, d in zip(host_flat, device_flat):
        assert abs(h - d) < 1e-9


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_compute_gradients_device_indices_match_layout(ff_cls, two_mols):
    """Device3DResult.atom_starts / mol_indices / conf_indices match the host layout."""
    ff = ff_cls(two_mols)
    result = ff.compute_gradients(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    atom_starts = result.atom_starts.torch().tolist()
    mol_indices = result.mol_indices.torch().tolist()
    conf_indices = result.conf_indices.torch().tolist()

    expected_n_conformers = sum(m.GetNumConformers() for m in two_mols)
    assert len(mol_indices) == expected_n_conformers
    assert len(conf_indices) == expected_n_conformers
    assert len(atom_starts) == expected_n_conformers + 1
    assert atom_starts[0] == 0

    cursor = 0
    idx = 0
    for mol_idx, mol in enumerate(two_mols):
        natoms = mol.GetNumAtoms()
        for conf_idx in range(mol.GetNumConformers()):
            assert mol_indices[idx] == mol_idx
            assert conf_indices[idx] == conf_idx
            cursor += natoms
            assert atom_starts[idx + 1] == cursor
            idx += 1


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_minimize_device_returns_device3d_with_optimized_positions(ff_cls, two_mols):
    """minimize(output=DEVICE) returns Device3DResult; values reflect updated coords."""
    ff = ff_cls(two_mols)
    pos_initial = ff.compute_gradients(output=CoordinateOutput.DEVICE).values.torch().clone()
    result = ff.minimize(maxIters=15, output=CoordinateOutput.DEVICE)
    assert isinstance(result, Device3DResult)
    torch.cuda.synchronize()
    pos_after = result.values.torch()
    assert pos_after.shape == pos_initial.shape
    assert (pos_after != pos_initial).any().item()


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_minimize_device_energies_match_host(ff_cls, two_mols):
    mols_host = [Chem.Mol(m) for m in two_mols]
    mols_device = [Chem.Mol(m) for m in two_mols]

    ff_host = ff_cls(mols_host)
    energies_host_nested, _ = ff_host.minimize(maxIters=20)
    energies_host = _flatten(energies_host_nested)

    ff_device = ff_cls(mols_device)
    result = ff_device.minimize(maxIters=20, output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    energies_device = result.energies.torch().tolist()

    assert len(energies_device) == len(energies_host)
    for h, d in zip(energies_host, energies_device):
        assert abs(h - d) < 1e-3, f"host {h} vs device {d}"


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_compute_energy_device_two_consecutive_results_independent(ff_cls, two_mols):
    """Holding a DEVICE energy result across a second compute_energy(DEVICE) must not corrupt it.

    Regression test for the scratch-buffer aliasing bug where the wrapper's persistent buffer
    was returned borrowed and then zeroed on the next call.
    """
    ff = ff_cls(two_mols)
    first = ff.compute_energy(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    first_snapshot = first.energies.torch().clone()

    second = ff.compute_energy(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    # The first result's energies must not have been clobbered by the second call.
    assert torch.equal(first.energies.torch(), first_snapshot)
    # Sanity: both results have the same numerical content (input positions unchanged).
    assert torch.allclose(first.energies.torch(), second.energies.torch())


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_compute_gradients_device_two_consecutive_results_independent(ff_cls, two_mols):
    """Holding a DEVICE gradient result across a second compute_gradients(DEVICE) must not corrupt it."""
    ff = ff_cls(two_mols)
    first = ff.compute_gradients(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    first_snapshot = first.values.torch().clone()

    second = ff.compute_gradients(output=CoordinateOutput.DEVICE)
    torch.cuda.synchronize()
    assert torch.equal(first.values.torch(), first_snapshot)
    assert torch.allclose(first.values.torch(), second.values.torch())


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_minimize_device_rejects_cross_gpu_target(ff_cls, two_mols):
    """The wrapper is single-GPU; minimize(output=DEVICE, target_gpu=other) must raise."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires >= 2 GPUs to exercise cross-GPU rejection")
    with torch.cuda.device(0):
        ff = ff_cls(two_mols)
        with pytest.raises(Exception, match="does not support target_gpu"):
            ff.minimize(maxIters=5, output=CoordinateOutput.DEVICE, target_gpu=1)
