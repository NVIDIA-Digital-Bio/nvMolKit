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

import copy

import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from nvmolkit.conformerRmsd import GetConformerRMSMatrix


def _embed_mol(smiles, num_confs=10, seed=42):
    """Helper: create a molecule with conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def _rdkit_rmsd_matrix(mol, prealigned=False):
    """Helper: compute reference RMSD matrix using RDKit."""
    return list(AllChem.GetConformerRMSMatrix(mol, prealigned=prealigned))


@pytest.mark.parametrize("smiles", ["CCCCCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
def test_rmsd_matches_rdkit(smiles):
    """GPU RMSD matrix matches RDKit reference within tolerance."""
    mol = _embed_mol(smiles, num_confs=20)
    no_h = Chem.RemoveHs(mol)

    # RDKit's GetConformerRMSMatrix(prealigned=False) modifies conformer
    # coordinates in-place during sequential alignment, so use a deep copy
    # to preserve the original coordinates for the GPU computation.
    rdkit_rms = _rdkit_rmsd_matrix(copy.deepcopy(no_h), prealigned=False)
    gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy().tolist()

    assert len(gpu_rms) == len(rdkit_rms), (
        f"Length mismatch: GPU={len(gpu_rms)}, RDKit={len(rdkit_rms)}"
    )

    for i, (g, r) in enumerate(zip(gpu_rms, rdkit_rms)):
        assert abs(g - r) < 0.05, (
            f"RMSD mismatch at index {i}: GPU={g:.6f}, RDKit={r:.6f}, diff={abs(g - r):.6f}"
        )


@pytest.mark.parametrize("smiles", ["CCCCCC", "c1ccccc1"])
def test_rmsd_prealigned_matches_rdkit(smiles):
    """GPU RMSD matrix with prealigned=True matches RDKit reference."""
    mol = _embed_mol(smiles, num_confs=10)
    no_h = Chem.RemoveHs(mol)

    rdkit_rms = _rdkit_rmsd_matrix(no_h, prealigned=True)
    gpu_result = GetConformerRMSMatrix(no_h, prealigned=True)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy().tolist()

    assert len(gpu_rms) == len(rdkit_rms)
    for i, (g, r) in enumerate(zip(gpu_rms, rdkit_rms)):
        assert abs(g - r) < 0.01, (
            f"RMSD mismatch at index {i}: GPU={g:.6f}, RDKit={r:.6f}"
        )


def test_rmsd_large_conformer_set():
    """Test with a larger conformer set (typical production size)."""
    mol = _embed_mol("CCCCCCCCCC", num_confs=100)
    no_h = Chem.RemoveHs(mol)

    n = no_h.GetNumConformers()
    expected_pairs = n * (n - 1) // 2

    gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy()

    assert gpu_rms.shape[0] == expected_pairs
    assert np.all(np.isfinite(gpu_rms))
    assert np.all(gpu_rms >= 0.0)


def test_rmsd_two_conformers():
    """Minimal case: exactly two conformers."""
    mol = _embed_mol("CCCC", num_confs=2)
    no_h = Chem.RemoveHs(mol)

    rdkit_rms = _rdkit_rmsd_matrix(no_h)
    gpu_result = GetConformerRMSMatrix(no_h)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy()

    assert gpu_rms.shape[0] == 1
    assert abs(gpu_rms[0] - rdkit_rms[0]) < 0.05


def test_rmsd_rigid_molecule():
    """Rigid molecule (benzene) — all conformers should have near-zero RMSD."""
    mol = _embed_mol("c1ccccc1", num_confs=5)
    no_h = Chem.RemoveHs(mol)

    gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy()

    assert np.all(gpu_rms < 0.5), f"Rigid molecule should have small RMSD, got max={gpu_rms.max():.4f}"


def test_rmsd_explicit_stream():
    """Test execution on an explicit CUDA stream."""
    mol = _embed_mol("CCCCCC", num_confs=10)
    no_h = Chem.RemoveHs(mol)

    s = torch.cuda.Stream()
    gpu_result = GetConformerRMSMatrix(no_h, stream=s)
    s.synchronize()

    rdkit_rms = _rdkit_rmsd_matrix(copy.deepcopy(no_h))
    gpu_rms = gpu_result.numpy().tolist()

    for g, r in zip(gpu_rms, rdkit_rms):
        assert abs(g - r) < 0.05


def test_rmsd_invalid_input_none():
    """None molecule should raise ValueError."""
    with pytest.raises(ValueError, match="mol must not be None"):
        GetConformerRMSMatrix(None)


def test_rmsd_invalid_input_no_conformers():
    """Molecule with fewer than 2 conformers should raise ValueError."""
    mol = Chem.MolFromSmiles("CCO")
    with pytest.raises(ValueError, match="mol must have at least 2 conformers"):
        GetConformerRMSMatrix(mol)


def test_rmsd_invalid_stream_type():
    """Non-stream argument should raise TypeError."""
    mol = _embed_mol("CCCC", num_confs=2)
    no_h = Chem.RemoveHs(mol)
    with pytest.raises(TypeError):
        GetConformerRMSMatrix(no_h, stream=42)
