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

"""Tests for GPU-accelerated TFD calculation."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints

import nvmolkit.tfd as tfd

# Tolerance for comparing GPU vs RDKit results
TOLERANCE = 0.01


def generate_conformers(mol, num_confs, seed=42):
    """Generate conformers for a molecule using ETKDG."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


@pytest.fixture
def simple_mol_with_conformers():
    """Create a simple molecule with conformers."""
    mol = Chem.MolFromSmiles("CCCCC")  # n-pentane
    return generate_conformers(mol, 5)


@pytest.fixture
def multiple_mols_with_conformers():
    """Create multiple molecules with conformers."""
    smiles_list = ["CCCC", "CCCCC", "CCCCCC", "CCO", "CCCO"]
    mols = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = generate_conformers(mol, 3 + i, seed=42 + i)
            if mol.GetNumConformers() >= 2:
                mols.append(mol)
    return mols


class TestGetTFDMatrix:
    """Tests for single-molecule TFD calculation."""

    def test_basic_tfd_matrix(self, simple_mol_with_conformers):
        """Test basic TFD matrix computation."""
        mol = simple_mol_with_conformers
        num_conf = mol.GetNumConformers()

        result = tfd.GetTFDMatrix(mol)

        # Check size: C*(C-1)/2 for C conformers
        expected_size = num_conf * (num_conf - 1) // 2
        assert len(result) == expected_size

        # All TFD values should be non-negative
        for val in result:
            assert val >= 0.0

    def test_single_conformer_returns_empty(self):
        """Test that single conformer returns empty matrix."""
        mol = Chem.MolFromSmiles("CCCC")
        generate_conformers(mol, 1)
        assert mol.GetNumConformers() == 1

        result = tfd.GetTFDMatrix(mol)
        assert len(result) == 0

    def test_cpu_backend(self, simple_mol_with_conformers):
        """Test CPU backend gives same results as GPU."""
        mol = simple_mol_with_conformers

        gpu_result = tfd.GetTFDMatrix(mol, backend="gpu")
        cpu_result = tfd.GetTFDMatrix(mol, backend="cpu")

        assert len(gpu_result) == len(cpu_result)
        for gpu_val, cpu_val in zip(gpu_result, cpu_result):
            assert abs(gpu_val - cpu_val) < 1e-4

    def test_no_weights(self, simple_mol_with_conformers):
        """Test computation without weights."""
        mol = simple_mol_with_conformers

        with_weights = tfd.GetTFDMatrix(mol, useWeights=True)
        no_weights = tfd.GetTFDMatrix(mol, useWeights=False)

        # Both should have same size
        assert len(with_weights) == len(no_weights)

        # Values may differ when weights are disabled
        # Just verify computation completes

    def test_maxdev_spec(self, simple_mol_with_conformers):
        """Test with specific max deviation mode."""
        mol = simple_mol_with_conformers

        equal_result = tfd.GetTFDMatrix(mol, maxDev="equal")
        spec_result = tfd.GetTFDMatrix(mol, maxDev="spec")

        assert len(equal_result) == len(spec_result)

    def test_invalid_maxdev_raises(self, simple_mol_with_conformers):
        """Test that invalid maxDev raises error."""
        mol = simple_mol_with_conformers

        with pytest.raises(Exception):
            tfd.GetTFDMatrix(mol, maxDev="invalid")

    def test_invalid_backend_raises(self, simple_mol_with_conformers):
        """Test that invalid backend raises error."""
        mol = simple_mol_with_conformers

        with pytest.raises(Exception):
            tfd.GetTFDMatrix(mol, backend="invalid")


class TestGetTFDMatrices:
    """Tests for batch TFD calculation."""

    def test_batch_processing(self, multiple_mols_with_conformers):
        """Test batch processing of multiple molecules."""
        mols = multiple_mols_with_conformers

        results = tfd.GetTFDMatrices(mols)

        assert len(results) == len(mols)

        for i, (mol, result) in enumerate(zip(mols, results)):
            num_conf = mol.GetNumConformers()
            expected_size = num_conf * (num_conf - 1) // 2
            assert len(result) == expected_size, f"Molecule {i} has wrong result size"

    def test_empty_input(self):
        """Test with empty molecule list."""
        results = tfd.GetTFDMatrices([])
        assert len(results) == 0

    def test_batch_vs_individual(self, multiple_mols_with_conformers):
        """Test that batch results match individual processing."""
        mols = multiple_mols_with_conformers

        batch_results = tfd.GetTFDMatrices(mols)
        individual_results = [tfd.GetTFDMatrix(mol) for mol in mols]

        assert len(batch_results) == len(individual_results)

        for batch, individual in zip(batch_results, individual_results):
            assert len(batch) == len(individual)
            for b, ind in zip(batch, individual):
                assert abs(b - ind) < 1e-4


class TestGpuResidentOutput:
    """Tests for GPU-resident TFD output."""

    def test_gpu_result_basic(self, simple_mol_with_conformers):
        """Test basic GPU-resident output."""
        mol = simple_mol_with_conformers

        result = tfd.GetTFDMatrixGpu(mol)

        # Should be an AsyncGpuResult
        assert hasattr(result, "torch")
        assert hasattr(result, "numpy")

        # Get as torch tensor
        tensor = result.torch()
        torch.cuda.synchronize()

        assert tensor.device.type == "cuda"
        num_conf = mol.GetNumConformers()
        expected_size = num_conf * (num_conf - 1) // 2
        assert tensor.shape[0] == expected_size

    def test_gpu_matrices_result(self, multiple_mols_with_conformers):
        """Test GPU-resident batch output."""
        mols = multiple_mols_with_conformers

        result = tfd.GetTFDMatricesGpu(mols)

        # Check metadata
        assert len(result.conformer_counts) == len(mols)
        assert len(result.output_starts) == len(mols) + 1

        # Extract individual results
        extracted = result.extract_all()
        assert len(extracted) == len(mols)

        for i, (mol, tensor) in enumerate(zip(mols, extracted)):
            num_conf = mol.GetNumConformers()
            expected_size = num_conf * (num_conf - 1) // 2
            assert tensor.shape[0] == expected_size, f"Molecule {i} has wrong tensor size"

    def test_extract_molecule(self, multiple_mols_with_conformers):
        """Test extracting single molecule from batch result."""
        mols = multiple_mols_with_conformers

        result = tfd.GetTFDMatricesGpu(mols)

        for i, mol in enumerate(mols):
            extracted = result.extract_molecule(i)
            num_conf = mol.GetNumConformers()
            expected_size = num_conf * (num_conf - 1) // 2
            assert extracted.shape[0] == expected_size

    def test_extract_molecule_out_of_range(self, multiple_mols_with_conformers):
        """Test that out-of-range index raises error."""
        mols = multiple_mols_with_conformers
        result = tfd.GetTFDMatricesGpu(mols)

        with pytest.raises(IndexError):
            result.extract_molecule(-1)

        with pytest.raises(IndexError):
            result.extract_molecule(len(mols))

    def test_to_lists(self, multiple_mols_with_conformers):
        """Test converting GPU result to Python lists."""
        mols = multiple_mols_with_conformers

        gpu_result = tfd.GetTFDMatricesGpu(mols)
        lists = gpu_result.to_lists()

        # Compare with direct computation
        direct = tfd.GetTFDMatrices(mols)

        assert len(lists) == len(direct)
        for gpu_list, direct_list in zip(lists, direct):
            assert len(gpu_list) == len(direct_list)
            for g, d in zip(gpu_list, direct_list):
                assert abs(g - d) < 1e-4


class TestCompareWithRDKit:
    """Tests comparing nvMolKit TFD with RDKit TFD."""

    def test_simple_chain_molecule(self):
        """Test simple chain molecule against RDKit."""
        mol = Chem.MolFromSmiles("CCCCC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"nvMolKit={nv}, RDKit={rd}"

    def test_no_weights_comparison(self):
        """Test without weights against RDKit."""
        mol = Chem.MolFromSmiles("CCCCCC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=False, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE

    def test_ring_molecule(self):
        """Test ring molecule (multi-quartet) against RDKit."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"nvMolKit={nv}, RDKit={rd}"

    def test_ring_with_substituent(self):
        """Test molecule with both ring and non-ring torsions against RDKit."""
        mol = Chem.MolFromSmiles("c1ccccc1CC")  # ethylbenzene
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"nvMolKit={nv}, RDKit={rd}"

    @pytest.mark.parametrize("backend", ["gpu", "cpu"])
    @pytest.mark.parametrize(
        "smiles",
        ["CCCCC", "CC(C)CC"],
    )
    def test_add_hs(self, backend, smiles):
        """Test molecules with explicit hydrogens against RDKit."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal", backend=backend)
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"AddHs({smiles}), backend={backend}: nvMolKit={nv}, RDKit={rd}"

    @pytest.mark.parametrize("backend", ["gpu", "cpu"])
    def test_symmetric_molecule_gpu_cpu_consistency(self, backend):
        """Test branched molecule gives consistent results across backends."""
        mol = Chem.MolFromSmiles("CC(C)CC")  # isopentane
        generate_conformers(mol, 4)

        result = tfd.GetTFDMatrix(mol, backend=backend)
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol)

        assert len(result) == len(rdkit_result)

        for nv, rd in zip(result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"backend={backend}: nvMolKit={nv}, RDKit={rd}"

    @pytest.mark.parametrize("backend", ["gpu", "cpu"])
    @pytest.mark.parametrize("symm_radius", [0, 1, 3])
    def test_symm_radius(self, backend, symm_radius):
        """Test different symmRadius values against RDKit on both backends."""
        mol = Chem.MolFromSmiles("CC(C)CC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(
            mol,
            symmRadius=symm_radius,
            backend=backend,
        )
        rdkit_result = TorsionFingerprints.GetTFDMatrix(
            mol,
            symmRadius=symm_radius,
        )

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"symmRadius={symm_radius}, backend={backend}: nvMolKit={nv}, RDKit={rd}"

    @pytest.mark.parametrize("backend", ["gpu", "cpu"])
    def test_ignore_colinear_bonds_false(self, backend):
        """Test ignoreColinearBonds=False against RDKit on both backends."""
        mol = Chem.MolFromSmiles("CCCC#CCC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(
            mol,
            ignoreColinearBonds=False,
            backend=backend,
        )
        rdkit_result = TorsionFingerprints.GetTFDMatrix(
            mol,
            ignoreColinearBonds=False,
        )

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"ignoreColinearBonds=False, backend={backend}: nvMolKit={nv}, RDKit={rd}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_ethane_no_torsions(self):
        """Test ethane (terminal bonds only, no rotatable bonds)."""
        mol = Chem.MolFromSmiles("CC")
        generate_conformers(mol, 3)

        result = tfd.GetTFDMatrix(mol)

        # Should return zeros
        for val in result:
            assert val == 0.0

    def test_large_molecule(self):
        """Test larger molecule."""
        # Decane
        mol = Chem.MolFromSmiles("CCCCCCCCCC")
        generate_conformers(mol, 5)

        result = tfd.GetTFDMatrix(mol)

        num_conf = mol.GetNumConformers()
        expected_size = num_conf * (num_conf - 1) // 2
        assert len(result) == expected_size

        for val in result:
            assert 0.0 <= val <= 1.0  # TFD should be normalized

    def test_invalid_molecule_raises(self):
        """Test that None molecule raises error."""
        with pytest.raises(Exception):
            tfd.GetTFDMatrix(None)

    def test_invalid_molecule_in_batch_raises(self, simple_mol_with_conformers):
        """Test that None in batch raises error."""
        mols = [simple_mol_with_conformers, None]

        with pytest.raises(Exception):
            tfd.GetTFDMatrices(mols)
