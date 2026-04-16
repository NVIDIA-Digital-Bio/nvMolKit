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

import copy

import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from nvmolkit.conformerRmsd import GetConformerRMSMatrix, GetConformerRMSMatrixBatch


def _numpy_kabsch_rmsd(p, q):
    """Independent Kabsch RMSD using numpy SVD (gold reference)."""
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    H = p_c.T @ q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(H))
    S[-1] *= d if d != 0.0 else 1.0
    Sp = np.sum(p_c**2)
    Sq = np.sum(q_c**2)
    return np.sqrt(max((Sp + Sq - 2.0 * np.sum(S)) / len(p), 0.0))


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


def _numpy_rmsd_matrix(mol, prealigned=False):
    """Compute full RMSD matrix using numpy Kabsch (independent reference)."""
    confs = mol.GetConformers()
    n = len(confs)
    coords = [np.array(c.GetPositions()) for c in confs]
    result = []
    for i in range(n):
        for j in range(i):
            if prealigned:
                diff = coords[i] - coords[j]
                rmsd = np.sqrt(np.sum(diff**2) / len(coords[i]))
            else:
                rmsd = _numpy_kabsch_rmsd(coords[i], coords[j])
            result.append(rmsd)
    return result


@pytest.mark.parametrize("smiles", ["CCCCCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
def test_rmsd_matches_numpy_kabsch(smiles):
    """GPU RMSD matrix matches numpy Kabsch SVD reference within tolerance."""
    mol = _embed_mol(smiles, num_confs=20)
    no_h = Chem.RemoveHs(mol)

    ref_rms = _numpy_rmsd_matrix(no_h, prealigned=False)
    gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy().tolist()

    assert len(gpu_rms) == len(ref_rms), f"Length mismatch: GPU={len(gpu_rms)}, ref={len(ref_rms)}"

    for i, (g, r) in enumerate(zip(gpu_rms, ref_rms)):
        assert abs(g - r) < 0.01, f"RMSD mismatch at index {i}: GPU={g:.6f}, numpy={r:.6f}, diff={abs(g - r):.6f}"


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
        assert abs(g - r) < 0.01, f"RMSD mismatch at index {i}: GPU={g:.6f}, RDKit={r:.6f}"


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

    ref_rms = _numpy_rmsd_matrix(no_h)
    gpu_result = GetConformerRMSMatrix(no_h)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy()

    assert gpu_rms.shape[0] == 1
    assert abs(gpu_rms[0] - ref_rms[0]) < 0.01


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

    ref_rms = _numpy_rmsd_matrix(no_h)
    gpu_rms = gpu_result.numpy().tolist()

    for g, r in zip(gpu_rms, ref_rms):
        assert abs(g - r) < 0.01


def test_rmsd_invalid_input_none():
    """None molecule should raise ValueError."""
    with pytest.raises(ValueError, match="mol must not be None"):
        GetConformerRMSMatrix(None)


def test_rmsd_fewer_than_two_conformers():
    """Molecule with fewer than 2 conformers returns empty result."""
    mol = Chem.MolFromSmiles("CCO")
    result = GetConformerRMSMatrix(mol)
    assert result.numpy().shape[0] == 0


def test_rmsd_invalid_stream_type():
    """Non-stream argument should raise TypeError."""
    mol = _embed_mol("CCCC", num_confs=2)
    no_h = Chem.RemoveHs(mol)
    with pytest.raises(TypeError):
        GetConformerRMSMatrix(no_h, stream=42)


# ---------------------------------------------------------------------------
# Batch API tests
# ---------------------------------------------------------------------------


def test_batch_matches_single():
    """Batch results match the single-molecule API for each molecule."""
    smiles_list = ["CCCCCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=10)) for s in smiles_list]

    batch_results = GetConformerRMSMatrixBatch(mols, prealigned=False)
    torch.cuda.synchronize()

    for mol, batch_result in zip(mols, batch_results):
        single_result = GetConformerRMSMatrix(mol, prealigned=False)
        torch.cuda.synchronize()

        batch_rms = batch_result.numpy()
        single_rms = single_result.numpy()
        np.testing.assert_allclose(batch_rms, single_rms, atol=1e-10, err_msg="Batch and single-mol results differ")


def test_batch_mixed_conformer_counts():
    """Batch handles molecules with different conformer counts."""
    mol_many = Chem.RemoveHs(_embed_mol("CCCCCC", num_confs=20))
    mol_few = Chem.RemoveHs(_embed_mol("CC", num_confs=3))
    mol_one = Chem.RemoveHs(_embed_mol("CCO", num_confs=1))  # below threshold

    results = GetConformerRMSMatrixBatch([mol_many, mol_few, mol_one])
    torch.cuda.synchronize()

    n_many = mol_many.GetNumConformers()
    n_few = mol_few.GetNumConformers()

    assert results[0].numpy().shape[0] == n_many * (n_many - 1) // 2
    assert results[1].numpy().shape[0] == n_few * (n_few - 1) // 2
    assert results[2].numpy().shape[0] == 0


def test_batch_empty_list():
    """Empty input returns an empty list."""
    results = GetConformerRMSMatrixBatch([])
    assert results == []


def test_batch_prealigned_matches_single():
    """Batch prealigned=True path matches the single-molecule API."""
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=8)) for s in ["CCCCCC", "c1ccccc1"]]

    batch_results = GetConformerRMSMatrixBatch(mols, prealigned=True)
    torch.cuda.synchronize()

    for mol, batch_result in zip(mols, batch_results):
        single_result = GetConformerRMSMatrix(mol, prealigned=True)
        torch.cuda.synchronize()
        np.testing.assert_allclose(
            batch_result.numpy(),
            single_result.numpy(),
            atol=1e-10,
            err_msg="Batch prealigned and single-mol results differ",
        )


def test_batch_invalid_none():
    """None molecule in list raises ValueError."""
    mol = Chem.RemoveHs(_embed_mol("CCCC", num_confs=2))
    with pytest.raises(ValueError):
        GetConformerRMSMatrixBatch([mol, None])


def test_batch_explicit_stream():
    """Batch results are correct on an explicit CUDA stream."""
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=5)) for s in ["CCCC", "CCCCC"]]

    s = torch.cuda.Stream()
    results = GetConformerRMSMatrixBatch(mols, stream=s)
    s.synchronize()

    for mol, result in zip(mols, results):
        ref = _numpy_rmsd_matrix(mol)
        rms = result.numpy().tolist()
        for g, r in zip(rms, ref):
            assert abs(g - r) < 0.01


def test_rmsd_zero_atoms():
    """0-atom molecule with multiple conformers raises ValueError.

    nvMolKit intentionally diverges from RDKit here: RDKit returns [nan] for
    exactly 2 zero-atom conformers and raises ZeroDivisionError for 3+.
    nvMolKit raises ValueError consistently for all degenerate zero-atom inputs.
    Such molecules cannot be produced by standard RDKit embedding workflows.
    """
    mol = Chem.RWMol()
    mol.AddConformer(Chem.Conformer(0), assignId=True)
    mol.AddConformer(Chem.Conformer(0), assignId=True)
    with pytest.raises(ValueError):
        GetConformerRMSMatrix(mol.GetMol())
