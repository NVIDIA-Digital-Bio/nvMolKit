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

import os

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.ForceField import rdForceField as _rdForceField  # noqa: F401
from rdkit.Geometry import Point3D

from nvmolkit._mmff_bridge import capture_mmff_settings
from nvmolkit.batchedForcefield import MMFFBatchedForcefield


def load_reference_mol():
    mol2_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "test_data",
        "rdkit_smallmol_1.mol2",
    )
    if not os.path.exists(mol2_path):
        pytest.skip(f"Test data file not found: {mol2_path}")
    mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
    if mol is None:
        pytest.skip("Failed to load rdkit_smallmol_1.mol2")
    Chem.SanitizeMol(mol)
    return mol


def make_embedded_mol(smiles: str, num_confs: int = 1, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def make_fragmented_mol():
    mol = make_embedded_mol("CC.CC")
    conf = mol.GetConformer()
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) != 2:
        raise AssertionError("Expected two fragments for interfragment interaction test")
    anchor = conf.GetAtomPosition(fragments[0][0])
    moved = conf.GetAtomPosition(fragments[1][0])
    shift = Point3D(anchor.x - moved.x + 2.0, anchor.y - moved.y, anchor.z - moved.z)
    for atom_idx in fragments[1]:
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, Point3D(pos.x + shift.x, pos.y + shift.y, pos.z + shift.z))
    return mol


def clone_mols(molecules):
    return [Chem.Mol(mol) for mol in molecules]


def make_rdkit_mmff_properties(mol, settings: dict | None = None):
    settings = {} if settings is None else settings
    variant = settings.get("variant", "MMFF94")
    mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if mmff_props is None:
        raise ValueError("RDKit could not create MMFF properties for molecule")
    mmff_props.SetMMFFVariant(variant)
    mmff_props.SetMMFFDielectricConstant(settings.get("dielectric_constant", 1.0))
    mmff_props.SetMMFFDielectricModel(settings.get("dielectric_model", 1))
    mmff_props.SetMMFFBondTerm(settings.get("bond_term", True))
    mmff_props.SetMMFFAngleTerm(settings.get("angle_term", True))
    mmff_props.SetMMFFStretchBendTerm(settings.get("stretch_bend_term", True))
    mmff_props.SetMMFFOopTerm(settings.get("oop_term", True))
    mmff_props.SetMMFFTorsionTerm(settings.get("torsion_term", True))
    mmff_props.SetMMFFVdWTerm(settings.get("vdw_term", True))
    mmff_props.SetMMFFEleTerm(settings.get("ele_term", True))
    return capture_mmff_settings(mmff_props, settings)


def make_rdkit_mmff_forcefield(
    mol,
    properties=None,
    conf_id: int = -1,
    nonBondedThreshold: float = 100.0,
    ignoreInterfragInteractions: bool = True,
):
    mmff_props = make_rdkit_mmff_properties(mol) if properties is None else properties
    return rdForceFieldHelpers.MMFFGetMoleculeForceField(
        mol,
        mmff_props,
        nonBondedThresh=nonBondedThreshold,
        confId=conf_id,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )


def get_mmff_reference_energy_and_grad(
    mol,
    properties=None,
    conf_id: int = -1,
    nonBondedThreshold: float = 100.0,
    ignoreInterfragInteractions: bool = True,
    configure_forcefield=None,
):
    ff = make_rdkit_mmff_forcefield(
        mol,
        properties=properties,
        conf_id=conf_id,
        nonBondedThreshold=nonBondedThreshold,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )
    if configure_forcefield is not None:
        configure_forcefield(ff)
    return ff.CalcEnergy(), list(ff.CalcGrad())


def assert_energy_and_gradient_close(got_energy, want_energy, got_grad, want_grad):
    assert got_energy == pytest.approx(want_energy, rel=1e-5, abs=1e-5)
    assert got_grad == pytest.approx(want_grad, rel=1e-4, abs=1e-4)


def assert_single_batched_matches_rdkit(
    mol,
    properties=None,
    nonBondedThreshold: float = 100.0,
    ignoreInterfragInteractions: bool = True,
    configure_batch=None,
    configure_rdkit=None,
):
    nvmolkit_mol = Chem.Mol(mol)
    ff = MMFFBatchedForcefield(
        [nvmolkit_mol],
        properties=properties,
        nonBondedThreshold=nonBondedThreshold,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )
    if configure_batch is not None:
        configure_batch(ff[0])
    got_energy = ff.compute_energy()[0][0]
    got_grad = ff.compute_gradients()[0][0]
    want_energy, want_grad = get_mmff_reference_energy_and_grad(
        Chem.Mol(mol),
        properties=properties,
        nonBondedThreshold=nonBondedThreshold,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
        configure_forcefield=configure_rdkit,
    )
    assert_energy_and_gradient_close(got_energy, want_energy, got_grad, want_grad)


def test_mmff_batched_forcefield_matches_rdkit():
    assert_single_batched_matches_rdkit(load_reference_mol())


def test_mmff_batched_forcefield_batch_matches_single():
    mols = [load_reference_mol(), load_reference_mol()]

    batch_ff = MMFFBatchedForcefield(clone_mols(mols))
    batch_energies = batch_ff.compute_energy()
    batch_grads = batch_ff.compute_gradients()

    single_energies = []
    single_grads = []
    for mol in clone_mols(mols):
        single_ff = MMFFBatchedForcefield([mol])
        single_energies.append(single_ff.compute_energy()[0][0])
        single_grads.append(single_ff.compute_gradients()[0][0])

    for mol_idx in range(len(mols)):
        assert batch_energies[mol_idx][0] == pytest.approx(single_energies[mol_idx], rel=1e-5, abs=1e-5)
        assert batch_grads[mol_idx][0] == pytest.approx(single_grads[mol_idx], rel=1e-4, abs=1e-4)


@pytest.mark.parametrize(
    ("mol_factory", "property_settings", "non_bonded_threshold", "ignore_interfrag_interactions"),
    [
        pytest.param(
            lambda: make_embedded_mol("CC(=O)NC"),
            {"variant": "MMFF94s"},
            100.0,
            True,
            id="variant-mmff94s",
        ),
        pytest.param(
            load_reference_mol,
            {"dielectric_constant": 2.5, "dielectric_model": 2},
            100.0,
            True,
            id="dielectric-settings",
        ),
        pytest.param(
            load_reference_mol,
            {
                "bond_term": False,
                "angle_term": False,
                "stretch_bend_term": False,
                "oop_term": False,
                "torsion_term": False,
            },
            100.0,
            True,
            id="term-toggles",
        ),
        pytest.param(
            make_fragmented_mol,
            None,
            25.0,
            False,
            id="interfragment-interactions",
        ),
    ],
)
def test_mmff_batched_forcefield_properties_match_rdkit(
    mol_factory, property_settings, non_bonded_threshold, ignore_interfrag_interactions
):
    mol = mol_factory()
    properties = None if property_settings is None else make_rdkit_mmff_properties(mol, property_settings)
    assert_single_batched_matches_rdkit(
        mol,
        properties=properties,
        nonBondedThreshold=non_bonded_threshold,
        ignoreInterfragInteractions=ignore_interfrag_interactions,
    )


def test_mmff_batched_forcefield_per_molecule_properties_match_rdkit():
    mols = [make_embedded_mol("CCO"), make_fragmented_mol()]
    properties = [
        make_rdkit_mmff_properties(mols[0], {"dielectric_constant": 3.0, "dielectric_model": 2}),
        make_rdkit_mmff_properties(mols[1]),
    ]
    non_bonded_thresholds = [100.0, 20.0]
    ignore_interfrag_interactions = [True, False]

    ff = MMFFBatchedForcefield(
        clone_mols(mols),
        properties=properties,
        nonBondedThreshold=non_bonded_thresholds,
        ignoreInterfragInteractions=ignore_interfrag_interactions,
    )
    got_energies = ff.compute_energy()
    got_grads = ff.compute_gradients()

    for idx, (mol, prop) in enumerate(zip(mols, properties)):
        want_energy, want_grad = get_mmff_reference_energy_and_grad(
            Chem.Mol(mol),
            properties=prop,
            nonBondedThreshold=non_bonded_thresholds[idx],
            ignoreInterfragInteractions=ignore_interfrag_interactions[idx],
        )
        assert_energy_and_gradient_close(got_energies[idx][0], want_energy, got_grads[idx][0], want_grad)


def test_mmff_batched_forcefield_multi_conformer_matches_rdkit():
    mol = make_embedded_mol("CCCO", num_confs=3)
    ff = MMFFBatchedForcefield([Chem.Mol(mol)])

    got_energies = ff.compute_energy()
    got_grads = ff.compute_gradients()

    assert len(got_energies) == 1
    assert len(got_energies[0]) == 3
    assert len(got_grads) == 1
    assert len(got_grads[0]) == 3

    for conf_idx, conf in enumerate(mol.GetConformers()):
        want_energy, want_grad = get_mmff_reference_energy_and_grad(
            Chem.Mol(mol), conf_id=conf.GetId()
        )
        assert_energy_and_gradient_close(
            got_energies[0][conf_idx], want_energy, got_grads[0][conf_idx], want_grad
        )


def test_mmff_batched_forcefield_lazy_build_and_rebuild():
    mol = make_embedded_mol("CCO")
    ff = MMFFBatchedForcefield([Chem.Mol(mol)])

    assert ff._native_ff is None
    assert ff._dirty is True

    first_energy = ff.compute_energy()[0][0]
    first_native = ff._native_ff

    assert first_native is not None
    assert ff._dirty is False
    assert ff.compute_energy()[0][0] == pytest.approx(first_energy, rel=1e-5, abs=1e-5)
    assert ff._native_ff is first_native

    ff[0].add_distance_constraint(0, 2, True, 0.2, 0.4, 25.0)
    assert ff._dirty is True

    ff.rebuild()
    assert ff._dirty is False
    assert ff._native_ff is not first_native


def test_mmff_batched_forcefield_invalid_indices():
    ff = MMFFBatchedForcefield([make_embedded_mol("CCO")])

    with pytest.raises(IndexError, match="Batch element index"):
        ff[1]

    with pytest.raises(IndexError, match="Atom index"):
        ff[0].add_distance_constraint(0, 99, False, 0.0, 1.0, 10.0)


def test_mmff_distance_constraint_matches_rdkit():
    mol = make_embedded_mol("CCO")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_distance_constraint(0, 2, False, 0.0, 1.5, 25.0),
        configure_rdkit=lambda ff: ff.MMFFAddDistanceConstraint(0, 2, False, 0.0, 1.5, 25.0),
    )


def test_mmff_distance_relative_constraint_matches_rdkit():
    mol = make_embedded_mol("CCO")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_distance_constraint(0, 2, True, 0.3, 0.6, 15.0),
        configure_rdkit=lambda ff: ff.MMFFAddDistanceConstraint(0, 2, True, 0.3, 0.6, 15.0),
    )


def test_mmff_position_constraint_matches_rdkit_reference_pose():
    mol = make_embedded_mol("CCO")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_position_constraint(0, 0.1, 50.0),
        configure_rdkit=lambda ff: ff.MMFFAddPositionConstraint(0, 0.1, 50.0),
    )


def test_mmff_angle_constraint_matches_rdkit():
    mol = make_embedded_mol("CCC")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_angle_constraint(0, 1, 2, True, 5.0, 10.0, 20.0),
        configure_rdkit=lambda ff: ff.MMFFAddAngleConstraint(0, 1, 2, True, 5.0, 10.0, 20.0),
    )


def test_mmff_torsion_constraint_matches_rdkit():
    mol = make_embedded_mol("CCCC")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_torsion_constraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
        configure_rdkit=lambda ff: ff.MMFFAddTorsionConstraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
    )


def test_mmff_mixed_properties_and_constraints_batch_matches_rdkit():
    mols = [make_embedded_mol("CCO"), make_embedded_mol("CCCC")]
    properties = [
        make_rdkit_mmff_properties(mols[0], {"dielectric_constant": 2.0, "dielectric_model": 2}),
        make_rdkit_mmff_properties(mols[1], {"variant": "MMFF94s"}),
    ]
    ff = MMFFBatchedForcefield(clone_mols(mols), properties=properties)
    ff[0].add_distance_constraint(0, 2, True, 0.2, 0.5, 20.0)
    ff[1].add_torsion_constraint(0, 1, 2, 3, True, 10.0, 20.0, 8.0)

    got_energies = ff.compute_energy()
    got_grads = ff.compute_gradients()

    ref_specs = [
        lambda forcefield: forcefield.MMFFAddDistanceConstraint(0, 2, True, 0.2, 0.5, 20.0),
        lambda forcefield: forcefield.MMFFAddTorsionConstraint(0, 1, 2, 3, True, 10.0, 20.0, 8.0),
    ]
    for idx, (mol, prop, configure_forcefield) in enumerate(zip(mols, properties, ref_specs)):
        want_energy, want_grad = get_mmff_reference_energy_and_grad(
            Chem.Mol(mol),
            properties=prop,
            configure_forcefield=configure_forcefield,
        )
        assert_energy_and_gradient_close(got_energies[idx][0], want_energy, got_grads[idx][0], want_grad)
