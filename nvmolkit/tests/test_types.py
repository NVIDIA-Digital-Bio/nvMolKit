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

import gc

import pytest
import torch

from rdkit.Chem import MolFromSmiles

from nvmolkit.fingerprints import MorganFingerprintGenerator

def _get_fps(num_mols):
    generator = MorganFingerprintGenerator(radius=0, fpSize=2048)
    template = MolFromSmiles("CC")
    mols = [template] * num_mols

    result = generator.GetFingerprints(mols)
    torch.cuda.synchronize()
    return result

def test_async_gpu_result_release_frees_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    base_free, _ = torch.cuda.mem_get_info()

    num_mols = 210_000
    expected_bytes = num_mols * 2048 // 8
    fps = _get_fps(num_mols)
    torch.cuda.synchronize()


    free_after_alloc, _ = torch.cuda.mem_get_info()
    assert free_after_alloc < base_free
    assert free_after_alloc + expected_bytes <= base_free

    del fps
    gc.collect()
    torch.cuda.synchronize()

    free_post, _ = torch.cuda.mem_get_info()

    assert (free_post - free_after_alloc) >= expected_bytes

