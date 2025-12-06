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

def check_butina_correctness(hit_mat, clusts):
    seen = set()

    # Verify clusters are in descending size order
    for i in range(1, len(clusts)):
        assert len(clusts[i - 1]) >= len(clusts[i]), \
            f"Clusters not in descending size order: cluster {i-1} has size {len(clusts[i-1])} but cluster {i} has size {len(clusts[i])}"

    for clust_idx, clust in enumerate(clusts):
        assert len(clust) > 0, "Empty cluster found"
        
        # Verify no point is assigned to multiple clusters
        for item in clust:
            assert item not in seen, f"Point {item} assigned to multiple clusters"
            seen.add(item)
        
        # Verify valid Butina cluster: there exists a centroid that is neighbor of all other members
        valid_cluster = False
        for centroid in clust:
            all_neighbors = all(hit_mat[centroid, member] for member in clust if member != centroid)
            if all_neighbors:
                valid_cluster = True
                break
        assert valid_cluster, f"Cluster {clust_idx} has no valid centroid"
    
    assert len(seen) == hit_mat.shape[0]

@pytest.mark.parametrize("size,neighborlist_max_size", 
                         [(s, n) for s in (1, 10, 100, 1000) for n in (8, 16, 24, 32, 64, 128)])
def test_butina_clustering(size, neighborlist_max_size):
    n = size
    cutoff = 0.1
    np.random.seed(42)
    dists = np.random.rand(n, n)
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to('cuda')
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).torch()
    nvmol_clusts = [tuple(torch.argwhere(nvmol_res == i).flatten().tolist()) for i in range(nvmol_res.max() + 1)]

    check_butina_correctness(torch_dists <= cutoff, nvmol_clusts)

@pytest.mark.parametrize("neighborlist_max_size", [8, 16, 24, 32, 64, 128])
def test_butina_edge_one_cluster(neighborlist_max_size):
    n = 10
    cutoff = 100.0
    dists = np.random.rand(n, n)
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to('cuda')
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).torch()
    assert torch.all(nvmol_res == 0)

@pytest.mark.parametrize("neighborlist_max_size", [8, 16, 24, 32, 64, 128])
def test_butina_edge_n_clusters(neighborlist_max_size):
    n = 10
    cutoff = 1e-8
    dists = np.random.rand(n, n)
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to('cuda')
    torch_dists = torch.clip(torch_dists, min=0.01)
    torch_dists.fill_diagonal_(0)
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).torch()
    assert torch.all(nvmol_res.sort()[0] == torch.arange(10).to('cuda'))
