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
import numpy as np
from nvmolkit.clustering import butina

def check_butina_correctness(hit_mat, clusts, strict):
    hit_mat = hit_mat.clone()
    seen = set()
    
    # In relaxed mode, sort clusters by size (descending)
    if not strict:
        clusts = sorted(clusts, key=len, reverse=True)

    for clust in clusts:
        assert len(clust) > 0
        clust_size = len(clust)
        counts = hit_mat.sum(-1)
        assert clust_size == counts.max(), f"Cluster size {clust_size} doesn't match max available count {counts.max()}"
        for item in clust:
            assert item not in seen, f"Point {item} assigned to multiple clusters"
            seen.add(item)
            hit_mat[item, :] = False
            hit_mat[:, item] = False
    assert len(seen) == hit_mat.shape[0]

@pytest.mark.parametrize("size,enforce_strict", [(s, e) for s in (1, 10, 100, 1000) for e in (True, False)])
def test_butina_clustering(size, enforce_strict):
    n = size
    cutoff = 0.1
    np.random.seed(42)
    dists = np.random.rand(n, n)
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to('cuda')
    nvmol_res = butina(torch_dists, cutoff, enforce_strict_indexing=enforce_strict).torch()
    nvmol_clusts = [tuple(torch.argwhere(nvmol_res == i).flatten().tolist()) for i in range(nvmol_res.max() + 1)]

    check_butina_correctness(torch_dists <= cutoff, nvmol_clusts, enforce_strict)

def test_butina_edge_one_cluster():
    n = 10
    cutoff = 100.0
    dists = np.random.rand(n, n)
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to('cuda')
    nvmol_res = butina(torch_dists, cutoff, enforce_strict_indexing=True).torch()
    assert torch.all(nvmol_res == 0)

def test_butina_edge_n_clusters():
    n = 10
    cutoff = 1e-8
    dists = np.random.rand(n, n)
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to('cuda')
    torch_dists = torch.clip(torch_dists, min=0.01)
    torch_dists.fill_diagonal_(0)
    nvmol_res = butina(torch_dists, cutoff, enforce_strict_indexing=True).torch()
    assert torch.all(nvmol_res.sort()[0] == torch.arange(10).to('cuda'))
