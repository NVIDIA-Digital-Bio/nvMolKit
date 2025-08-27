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
import os
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

import nvmolkit.mmffOptimization as nvmolkit_mmff
from nvmolkit.types import HardwareOptions


@pytest.fixture
def mmff_test_mols(num_mols=5):
    """Load molecules from MMFF94_dative.sdf for testing.
    
    Args:
        num_mols: Number of molecules to load (default: 5)
    
    Returns:
        list: A list of RDKit molecules with conformers from the SDF file.
    """
    # Path from nvmolkit/tests/ to tests/test_data/
    sdf_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..',                 # Go up to project root
        'tests', 'test_data', 'MMFF94_dative.sdf'
    )
    
    if not os.path.exists(sdf_path):
        pytest.skip(f"Test data file not found: {sdf_path}")
    
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    molecules = []
    
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if i >= num_mols:  # Load only requested number of molecules
            break
        molecules.append(mol)
    
    if len(molecules) < num_mols:
        pytest.skip(f"Expected {num_mols} molecules, but found only {len(molecules)} in {sdf_path}")
    
    return molecules


def create_hard_copy_mols(molecules):
    """Create true hard copies of molecules with their conformers.
    
    Args:
        molecules: List of RDKit molecules to copy
        
    Returns:
        list: List of copied molecules with identical conformers
    """
    copied_mols = []
    for mol in molecules:
        # Create a new molecule from the original's structure
        copied_mol = Chem.Mol(mol)
        copied_mols.append(copied_mol)
    
    return copied_mols


def calculate_rdkit_mmff_energies(molecules):
    """Calculate MMFF energies using RDKit for all conformers of all molecules.
    
    Args:
        molecules: List of RDKit molecules with conformers
        
    Returns:
        list: List of lists containing energies for each molecule's conformers
    """
    all_energies = []
    
    for mol in molecules:
        mol_energies = []
        num_conformers = mol.GetNumConformers()
        
        if num_conformers == 0:
            all_energies.append([])
            continue
            
        # Optimize all conformers for this molecule using RDKit
        # The signature shows it's a method on the molecule object
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=200,
                                                               mmffVariant='MMFF94', nonBondedThresh=100.0)
        
        if results:
            for _, energy in results:
                mol_energies.append(energy)
        
        all_energies.append(mol_energies)
    
    return all_energies


def test_mmff_optimization_serial_vs_rdkit(mmff_test_mols):
    """Test nvMolKit MMFF optimization one molecule at a time against RDKit reference.
    
    This test compares the energy results when optimizing molecules individually
    using nvMolKit vs RDKit's MMFFOptimizeMoleculeConfs function.
    """
    # Create hard copies for fair comparison
    rdkit_mols = create_hard_copy_mols(mmff_test_mols)
    nvmolkit_mols = create_hard_copy_mols(mmff_test_mols)
    
    # Get RDKit reference energies
    rdkit_energies = calculate_rdkit_mmff_energies(rdkit_mols)
    
    # Get nvMolKit energies one molecule at a time (serial mode)
    nvmolkit_energies = []
    for mol in nvmolkit_mols:
        if mol.GetNumConformers() == 0:
            nvmolkit_energies.append([])
            continue
            
        # Call nvMolKit with single molecule
        mol_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
            [mol],
            maxIters=200,
            nonBondedThreshold=100.0,
        )
        nvmolkit_energies.extend(mol_energies)
    
    # Verify we have the same number of molecules
    assert len(rdkit_energies) == len(nvmolkit_energies), \
        f"Mismatch in number of molecules: RDKit={len(rdkit_energies)}, nvMolKit={len(nvmolkit_energies)}"
    
    # Compare energies for each molecule
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies), \
            f"Molecule {mol_idx}: conformer count mismatch: RDKit={len(rdkit_mol_energies)}, nvMolKit={len(nvmolkit_mol_energies)}"
        
        # Compare each conformer's energy with tolerance
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff
            
            assert rel_error < 1e-3, \
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: " \
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, " \
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"


@pytest.mark.parametrize("gpu_ids", [[0, 1], [0], [1]])
@pytest.mark.parametrize("batchesize", [0, 2, 5])
@pytest.mark.parametrize("batches_per_gpu", [1, 3])
def test_mmff_optimization_batch_vs_rdkit(mmff_test_mols, gpu_ids, batchesize, batches_per_gpu):
    """Test nvMolKit MMFF batch optimization against RDKit reference.
    
    This test compares the energy results when optimizing all molecules together
    in batch mode using nvMolKit vs individual RDKit optimization.
    """
    available_devices = torch.cuda.device_count()
    if available_devices == 1 and 1 in gpu_ids:
        pytest.skip("Test requires at least 2 GPUs for batch mode comparison")
    # Create hard copies for fair comparison
    rdkit_mols = create_hard_copy_mols(mmff_test_mols)
    nvmolkit_mols = create_hard_copy_mols(mmff_test_mols)
    
    # Get RDKit reference energies
    rdkit_energies = calculate_rdkit_mmff_energies(rdkit_mols)

    hardware_options = HardwareOptions(
        gpuIds=gpu_ids,
        batchSize=batchesize,
        batchesPerGpu=batches_per_gpu,
    )
    
    # Get nvMolKit energies in batch mode (all molecules at once)
    nvmolkit_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
        nvmolkit_mols,
        maxIters=200,
        nonBondedThreshold=100.0,
        hardwareOptions=hardware_options
    )
    
    # Verify we have the same number of molecules
    assert len(rdkit_energies) == len(nvmolkit_energies), \
        f"Mismatch in number of molecules: RDKit={len(rdkit_energies)}, nvMolKit={len(nvmolkit_energies)}"
    
    # Compare energies for each molecule
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies), \
            f"Molecule {mol_idx}: conformer count mismatch: RDKit={len(rdkit_mol_energies)}, nvMolKit={len(nvmolkit_mol_energies)}"
        
        # Compare each conformer's energy with tolerance
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff
            
            assert rel_error < 1e-3, \
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: " \
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, " \
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"


def test_mmff_optimization_empty_input():
    """Test nvMolKit MMFF optimization with empty input."""
    result = nvmolkit_mmff.MMFFOptimizeMoleculesConfs([])
    assert result == []


def test_mmff_optimization_invalid_input():
    """Test nvMolKit MMFF optimization with invalid input."""
    with pytest.raises(ValueError, match="Molecule at index 0 is None"):
        nvmolkit_mmff.MMFFOptimizeMoleculesConfs([None])


def test_mmff_optimization_oversized_atom_limit_interleaved():
    """Ensure an oversized (>256 atoms) molecule in batch raises an error."""
    small1 = Chem.MolFromSmiles('CCCCCC')  # 6 atoms
    small2 = Chem.MolFromSmiles('CCC')     # 3 atoms
    # Oversized straight-chain hydrocarbon
    big = Chem.MolFromSmiles('C' * 300)
    assert big.GetNumAtoms() > 256

    # Need conformers for MMFF
    rdDistGeom.EmbedMultipleConfs(small1, numConfs=1)
    rdDistGeom.EmbedMultipleConfs(small2, numConfs=1)
    rdDistGeom.EmbedMultipleConfs(big, numConfs=1)

    with pytest.raises(ValueError, match=r"maximum supported is 256"):
        nvmolkit_mmff.MMFFOptimizeMoleculesConfs([small1, big, small2])
