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

"""Molecule preparation helpers shared across nvMolKit benchmarks."""

from rdkit import Chem


def prep_mols(
    mols: list[Chem.Mol],
    *,
    add_hs: bool = True,
    sanitize: bool = True,
    clear_conformers: bool = True,
) -> list[Chem.Mol]:
    """Return new molecule copies prepared for conformer generation.

    For each input:
        - skip ``None`` entries,
        - optionally add explicit hydrogens,
        - optionally sanitize (no-op when already sanitized),
        - optionally drop existing conformers so timed runs start clean.

    Molecules that fail any step are dropped and a count is printed.
    """
    prepped: list[Chem.Mol] = []
    drop_count = 0
    for mol in mols:
        if mol is None:
            drop_count += 1
            continue
        try:
            current = Chem.AddHs(mol) if add_hs else Chem.Mol(mol)
            if sanitize:
                Chem.SanitizeMol(current)
            if clear_conformers:
                current.RemoveAllConformers()
            prepped.append(current)
        except Exception:
            drop_count += 1
    if drop_count > 0:
        print(f"  Dropped {drop_count} molecules during prep (None or sanitize failure)")
    return prepped


def clone_mols_with_conformers(mols: list[Chem.Mol]) -> list[Chem.RWMol]:
    """Deep-copy molecules including their conformers.

    Useful for benches whose timed routines mutate conformer state in place
    (such as ETKDG embedding or FF optimization), so each iteration sees a
    pristine input.
    """
    return [Chem.RWMol(mol) for mol in mols]
