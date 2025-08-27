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

import pytest
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.similarity import bulkCosineSimilarity as BulkCosineSimilarityNvMolKit
from nvmolkit.similarity import bulkTanimotoSimilarity as BulkTanimotoSimilarityNvMolKit


# --------------------------------
# Edge cases and failure tests.
# --------------------------------
@pytest.mark.parametrize("simtype", ["tanimoto", "cosine"])
def test_bulk_similarity_empty_input(simtype, one_hundred_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    fps = [fpgen.GetFingerprint(mol) for mol in one_hundred_mols[0:1]]
    if simtype == "tanimoto":
        nvmolkit_sims = BulkTanimotoSimilarityNvMolKit(fps[0], [])
    else:
        nvmolkit_sims = BulkCosineSimilarityNvMolKit(fps[0], [])
    assert len(nvmolkit_sims) == 0

@pytest.mark.parametrize("simtype", ["tanimoto", "cosine"])
def test_bulk_similarity_fp_mismatch(simtype, one_hundred_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=128)
    fpgen2 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256)
    fps = [fpgen.GetFingerprint(mol) for mol in one_hundred_mols[0:1]]
    fps2 = [fpgen2.GetFingerprint(mol) for mol in one_hundred_mols]
    if simtype == "tanimoto":
        with pytest.raises(ValueError):
            nvmolkit_sims = BulkTanimotoSimilarityNvMolKit(fps[0], fps2)
    else:
        with pytest.raises(ValueError):
            nvmolkit_sims = BulkCosineSimilarityNvMolKit(fps[0], fps2)


# --------------------------------
# Tanimoto similarity tests
# --------------------------------

def test_nvmolkit_bulk_tanimoto(one_hundred_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    fps = [fpgen.GetFingerprint(mol) for mol in one_hundred_mols]
    ref_sims = BulkTanimotoSimilarity(fps[0], fps)
    nvmolkit_sims = BulkTanimotoSimilarityNvMolKit(fps[0], fps)
    assert all(abs(a - b) < 1e-5 for a, b in zip(ref_sims, nvmolkit_sims))

# --------------------------------
# Cosine similarity tests
# --------------------------------

def test_nvmolkit_bulk_cosine(one_hundred_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    fps = [fpgen.GetFingerprint(mol) for mol in one_hundred_mols]
    ref_sims = BulkCosineSimilarity(fps[0], fps)
    nvmolkit_sims = BulkCosineSimilarityNvMolKit(fps[0], fps)
    assert all(abs(a - b) < 1e-5 for a, b in zip(ref_sims, nvmolkit_sims))

