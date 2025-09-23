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
import torch
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.similarity import crossCosineSimilarity, crossTanimotoSimilarity
from nvmolkit.fingerprints import MorganFingerprintGenerator


# --------------------------------
# Edge cases and failure tests.
# --------------------------------
@pytest.mark.parametrize("simtype", ["tanimoto", "cosine"])
def test_cross_similarity_fp_mismatch(simtype, size_limited_mols):
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=128)
    nvmolkit_fpgen2 = MorganFingerprintGenerator(radius=3, fpSize=256)
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_cu2 = nvmolkit_fpgen2.GetFingerprints(size_limited_mols, num_threads=1)
    if simtype == "tanimoto":
        with pytest.raises(ValueError):
            nvmolkit_sims = crossTanimotoSimilarity(nvmolkit_fps_cu, nvmolkit_fps_cu2)
    else:
        with pytest.raises(ValueError):
            nvmolkit_sims = crossCosineSimilarity(nvmolkit_fps_cu, nvmolkit_fps_cu2)


# --------------------------------
# Tanimoto similarity tests
# --------------------------------

def test_nvmolkit_cross_tanimoto_similarity_from_nvmolkit_fp(size_limited_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_torch = nvmolkit_fps_cu.torch()
    assert nvmolkit_fps_torch.shape == (len(size_limited_mols), 1024 // 32)
    assert nvmolkit_fps_torch.device.type == 'cuda'
    ref_sims = torch.empty(len(fps), len(fps), dtype=torch.float64)
    for i in range(len(fps)):
        ref_sims[i] = torch.tensor(BulkTanimotoSimilarity(fps[i], fps))
    ref_sims = ref_sims.to('cuda')
    torch.cuda.synchronize()
    nvmolkit_sims = crossTanimotoSimilarity(nvmolkit_fps_torch).torch()

    torch.testing.assert_close(nvmolkit_sims, ref_sims)

    # Test that the same API can be used with the AsyncGpuResult directly.
    nvmolkit_sims_direct = crossTanimotoSimilarity(nvmolkit_fps_cu)
    torch.testing.assert_close(nvmolkit_sims_direct.torch(), ref_sims)



@pytest.mark.parametrize('nxmdims', ((1, 20), (20, 1), (20, 2), (29, 29)))
def test_nxm_cross_tanimoto_similarity_from_nvmolkit_fp(size_limited_mols, nxmdims):
    d1, d2 = nxmdims
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    assert d1 + d2 <= len(size_limited_mols), \
        f"Need enough molecules for {d1} x {d2} cross similarity, got {len(size_limited_mols)}"
    mols1 = size_limited_mols[:d1]
    mols2 = size_limited_mols[d1:d1+d2]

    fps1 = [fpgen.GetFingerprint(mol) for mol in mols1]
    fps2 = [fpgen.GetFingerprint(mol) for mol in mols2]

    nvmolkit_fps_cu1 = nvmolkit_fpgen.GetFingerprints(mols1, num_threads=1)
    nvmolkit_fps_cu2 = nvmolkit_fpgen.GetFingerprints(mols2, num_threads=1)

    nvmolkit_fps_torch1 = nvmolkit_fps_cu1.torch()
    nvmolkit_fps_torch2 = nvmolkit_fps_cu2.torch()

    assert nvmolkit_fps_torch1.shape == (len(mols1), 1024 // 32)
    assert nvmolkit_fps_torch1.device.type == 'cuda'
    assert nvmolkit_fps_torch2.shape == (len(mols2), 1024 // 32)
    assert nvmolkit_fps_torch2.device.type == 'cuda'


    ref_sims = torch.empty(len(fps1), len(fps2), dtype=torch.float64)
    for i in range(len(fps1)):
        ref_sims[i, :] = torch.tensor(BulkTanimotoSimilarity(fps1[i], fps2))
    ref_sims = ref_sims.to('cuda')
    torch.cuda.synchronize()
    nvmolkit_sims = crossTanimotoSimilarity(nvmolkit_fps_torch1, nvmolkit_fps_torch2).torch()
    torch.testing.assert_close(nvmolkit_sims, ref_sims)

@pytest.mark.parametrize('nxmdims', ((1, 20), (2, 10), (300, 300)))
def test_nxm_cross_tanimoto_similarity_from_packing(nxmdims):
    d1, d2 = nxmdims

    fps_1 = torch.randint(0, 2, (d1, 2048), dtype=torch.bool).to('cuda')
    fps_2 = torch.randint(0, 2, (d2, 2048), dtype=torch.bool).to('cuda')
    from nvmolkit.fingerprints import pack_fingerprint
    nvmolkit_fps_torch1 = pack_fingerprint(fps_1)
    nvmolkit_fps_torch2 = pack_fingerprint(fps_2)

    bitvects_a = []
    bitvects_b = []
    from rdkit.DataStructs import ExplicitBitVect
    for i in range(d1):
        bv = ExplicitBitVect(2048)
        for bit in range(2048):
            if fps_1[i, bit].item():
                bv.SetBit(bit)
        bitvects_a.append(bv)
    for i in range(d2):
        bv = ExplicitBitVect(2048)
        for bit in range(2048):
            if fps_2[i, bit].item():
                bv.SetBit(bit)
        bitvects_b.append(bv)

    ref_sims = torch.zeros(d1, d2, dtype=torch.float64)
    for i in range(d1):
        ref_sims[i, :] = torch.tensor(BulkTanimotoSimilarity(bitvects_a[i], bitvects_b))
    ref_sims = ref_sims.to('cuda')
    torch.cuda.synchronize()
    nvmolkit_sims = crossTanimotoSimilarity(nvmolkit_fps_torch1, nvmolkit_fps_torch2).torch()
    torch.testing.assert_close(nvmolkit_sims, ref_sims)
# --------------------------------
# Cosine similarity tests
# --------------------------------


def test_nvmolkit_cross_cosine_similarity_from_nvmolkit_fp(size_limited_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_torch = nvmolkit_fps_cu.torch()
    assert nvmolkit_fps_torch.shape == (len(size_limited_mols), 1024 // 32)
    assert nvmolkit_fps_torch.device.type == 'cuda'
    ref_sims = torch.empty(len(fps), len(fps), dtype=torch.float64)
    for i in range(len(fps)):
        ref_sims[i] = torch.tensor(BulkCosineSimilarity(fps[i], fps))
    ref_sims = ref_sims.to('cuda')
    torch.cuda.synchronize()
    nvmolkit_sims = crossCosineSimilarity(nvmolkit_fps_torch).torch()

    torch.testing.assert_close(nvmolkit_sims, ref_sims)

    # Test that the same API can be used with the AsyncGpuResult directly.
    nvmolkit_sims_direct = crossCosineSimilarity(nvmolkit_fps_cu)
    torch.testing.assert_close(nvmolkit_sims_direct.torch(), ref_sims)


@pytest.mark.parametrize('nxmdims', ((1, 20), (2, 10), (4, 5), (5, 4), (10, 2), (20, 1)))
def test_nxm_cross_cosine_similarity_from_nvmolkit_fp(size_limited_mols, nxmdims):
    d1, d2 = nxmdims
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    assert d1 + d2 <= len(size_limited_mols), \
        f"Need enough molecules for {d1} x {d2} cross similarity, got {len(size_limited_mols)}"
    mols1 = size_limited_mols[:d1]
    mols2 = size_limited_mols[d1:d1+d2]

    fps1 = [fpgen.GetFingerprint(mol) for mol in mols1]
    fps2 = [fpgen.GetFingerprint(mol) for mol in mols2]

    nvmolkit_fps_cu1 = nvmolkit_fpgen.GetFingerprints(mols1, num_threads=1)
    nvmolkit_fps_cu2 = nvmolkit_fpgen.GetFingerprints(mols2, num_threads=1)

    nvmolkit_fps_torch1 = nvmolkit_fps_cu1.torch()
    nvmolkit_fps_torch2 = nvmolkit_fps_cu2.torch()

    assert nvmolkit_fps_torch1.shape == (len(mols1), 1024 // 32)
    assert nvmolkit_fps_torch1.device.type == 'cuda'
    assert nvmolkit_fps_torch2.shape == (len(mols2), 1024 // 32)
    assert nvmolkit_fps_torch2.device.type == 'cuda'


    ref_sims = torch.empty(len(fps1), len(fps2), dtype=torch.float64)
    for i in range(len(fps1)):
        ref_sims[i, :] = torch.tensor(BulkCosineSimilarity(fps1[i], fps2))
    ref_sims = ref_sims.to('cuda')
    torch.cuda.synchronize()
    nvmolkit_sims = crossCosineSimilarity(nvmolkit_fps_torch1, nvmolkit_fps_torch2).torch()

    torch.testing.assert_close(nvmolkit_sims, ref_sims)
