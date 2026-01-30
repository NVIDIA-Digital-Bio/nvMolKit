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
    hit_mat = hit_mat.clone()
    seen = set()

    for i, clust in enumerate(clusts):
        assert len(clust) > 0, "Empty cluster found"
        clust_size = len(clust)

        if clust_size == 1:
            remaining_items = []
            for remaining_clust in clusts[i:]:
                assert len(remaining_clust) == 1, "Expected all remaining clusters to be singletons"
                remaining_items.append(remaining_clust[0])

            remaining_set = set(remaining_items)
            assert len(remaining_set) == len(remaining_items), "Duplicate items in singleton clusters"
            assert remaining_set.isdisjoint(seen), "Singleton item was already seen"
            seen.update(remaining_set)
            break
        counts = hit_mat.sum(-1)
        assert clust_size == counts.max(), f"Cluster size {clust_size} doesn't match max available count {counts.max()}"
        for item in clust:
            assert item not in seen, f"Point {item} assigned to multiple clusters"
            seen.add(item)
            hit_mat[item, :] = False
            hit_mat[:, item] = False
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

def test_butina_returns_centroids():
    n = 25
    cutoff = 0.2
    np.random.seed(123)
    dists = np.random.rand(n, n)
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to('cuda')
    clusters, centroids = butina(torch_dists, cutoff, return_centroids=True)
    clusters_tensor = clusters.torch()
    centroids_tensor = centroids.torch()

    num_clusters = int(clusters_tensor.max().item()) + 1
    assert centroids_tensor.numel() == num_clusters

    adjacency = torch_dists <= cutoff
    for cluster_id in range(num_clusters):
        centroid = int(centroids_tensor[cluster_id].item())
        assert clusters_tensor[centroid].item() == cluster_id
        members = torch.nonzero(clusters_tensor == cluster_id, as_tuple=False).flatten()
        for member in members:
            assert adjacency[centroid, member].item()
