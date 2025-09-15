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

from nvmolkit.fingerprints import MorganFingerprintGenerator, pack_fingerprint, unpack_fingerprint

def test_roundtrip_pack_unpack():
    # Create a test boolean tensor
    n_fps = 10
    fp_size = 128
    test_fp = torch.randint(0, 2, (n_fps, fp_size), dtype=torch.bool, device='cuda')

    # Pack it
    packed = pack_fingerprint(test_fp)
    assert packed.shape == (n_fps, fp_size // 32)
    assert packed.device.type == 'cuda'

    # Unpack it
    unpacked = unpack_fingerprint(packed)
    # Verify roundtrip
    torch.testing.assert_close(test_fp, unpacked)

def test_pack_unpack_uneven_size():
    fp_size = 127
    n_fps = 10
    test_fp = torch.randint(0, 2, (n_fps, fp_size), dtype=torch.bool, device='cpu')
    packed = pack_fingerprint(test_fp)
    assert packed.shape == (n_fps, 4)
    unpacked = unpack_fingerprint(packed)
    assert unpacked.shape == (n_fps, 128)
    torch.testing.assert_close(test_fp, unpacked[:, :fp_size])

@pytest.mark.parametrize('fpSize', (17, 8192))
def test_nvmolkit_fingerprint_throws_on_invalid_fpsize(fpSize, size_limited_mols):
    fpgen = MorganFingerprintGenerator(radius=3, fpSize=fpSize)
    with pytest.raises(Exception):
        fpgen.GetFingerprints(size_limited_mols)

def test_empty_input():
    fpgen = MorganFingerprintGenerator(radius=3, fpSize=2048)
    fps = fpgen.GetFingerprints([]).torch()
    torch.cuda.synchronize()
    assert fps.shape == (0, 2048 // 32)

def test_invalid_input():
    fpgen = MorganFingerprintGenerator(radius=3, fpSize=2048)
    with pytest.raises(Exception):
        fpgen.GetFingerprints([None])


@pytest.mark.parametrize('fpSize', (128, 1024, 2048))
@pytest.mark.parametrize('radius', (0, 1, 3, 5))
def test_nvmolkit_morgan_fingerprint(size_limited_mols, fpSize, radius):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]

    nvmolkit_fpgen = MorganFingerprintGenerator(radius=radius, fpSize=fpSize)
    nvmolkit_fps_torch =  nvmolkit_fpgen.GetFingerprints(size_limited_mols).torch()
    torch.cuda.synchronize()
    assert nvmolkit_fps_torch.device.type == 'cuda'
    want_n_rows = len(size_limited_mols)
    want_n_cols = fpSize / 32
    assert nvmolkit_fps_torch.shape == (want_n_rows, want_n_cols)
    for i in range(want_n_rows):
        ref_fp = fps[i]
        got_fp_row = nvmolkit_fps_torch[i, :]
        for j in range(fpSize):
            want_bit = ref_fp.GetBit(j)

            column = j // 32
            mask = 1 << (j % 32)

            got_bit = got_fp_row[column].item() & mask  != 0
            assert got_bit == want_bit
        # Now test that the unpacked fingerprint matches the original
        unpacked = unpack_fingerprint(got_fp_row.unsqueeze(0))
        assert unpacked.shape == (1, fpSize,)
        assert unpacked.device.type == 'cuda'
        assert unpacked.dtype == torch.bool
        torch.testing.assert_close(ref_fp.ToList(), unpacked.to(int).tolist()[0])
