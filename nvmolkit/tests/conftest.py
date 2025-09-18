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

import os

import pandas as pd
import pytest
from rdkit import Chem

@pytest.fixture
def one_hundred_smiles():
    """Load one hundred smiles from a CSV file.

    Returns:
        list: A list of one hundred SMILES strings.
    """
    path = os.path.join(os.path.dirname(__file__), "testdata", "smiles.csv")
    df = pd.read_csv(path)
    return df["smiles"].tolist()


@pytest.fixture
def one_hundred_mols(one_hundred_smiles):
    """Generate RDKit molecules from one hundred smiles.

    Args:
        one_hundred_smiles (list): A list of one hundred SMILES strings.

    Returns:
        list: A list of one hundred RDKit molecules.
    """
    return [Chem.MolFromSmiles(smi) for smi in one_hundred_smiles]


@pytest.fixture
def size_limited_mols(one_hundred_mols):
    """Generate RDKit molecules from one hundred smiles and discard any that have more than 128 atoms or bonds.

    Args:
        one_hundred_mols (list): A list of one hundred RDKit molecules.

    Returns:
        list: A list of RDKit molecules with at most 128 atoms and bonds. Up to 100 molecules are returned.
    """
    return [mol for mol in one_hundred_mols if mol.GetNumAtoms() <= 128 and mol.GetNumBonds() <= 128]
