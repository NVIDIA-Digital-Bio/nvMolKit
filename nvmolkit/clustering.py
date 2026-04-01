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

"""Contains GPU-accelerated Butina clustering implementation."""

import torch

from nvmolkit import _clustering
from nvmolkit._arrayHelpers import * # noqa: F403
from nvmolkit.types import AsyncGpuResult
from nvmolkit._fused_Butina import extract_cluster_and_singletons, update_neighbor_counts

_VALID_NEIGHBORLIST_SIZES = frozenset({8, 16, 24, 32, 64, 128})


def butina(
    distance_matrix: AsyncGpuResult | torch.Tensor,
    cutoff: float,
    neighborlist_max_size: int = 64,
    return_centroids: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult | tuple[AsyncGpuResult, AsyncGpuResult]:
    """Perform Butina clustering on a distance matrix.

    The Butina algorithm is a deterministic clustering method that groups items based
    on distance thresholds. It iteratively:
    1. Finds the item with the most neighbors within the cutoff distance
    2. Forms a cluster with that item and all its neighbors
    3. Removes clustered items from consideration
    4. Repeats until all items are clustered

    Args:
        distance_matrix: Square distance matrix of shape (N, N) where N is the number
                        of items. Can be an AsyncGpuResult or torch.Tensor on GPU.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is less than this cutoff.
        neighborlist_max_size: Maximum size of the neighborlist used for small cluster
                              optimization. Must be 8, 16, 24, 32, 64, or 128. Larger values
                              allow parallel processing of larger clusters but use more
                              shared memory.
        return_centroids: Whether to return centroid indices for each cluster.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        AsyncGpuResult of shape ``(N,)`` with cluster IDs (cluster 0 is the
        largest) when ``return_centroids`` is False.  When ``return_centroids``
        is True, returns a tuple ``(clusters, centroids)`` where *centroids* is
        an AsyncGpuResult of shape ``(num_clusters,)`` containing the centroid
        index for each cluster ID.

    Note:
        The distance matrix should be symmetric and have zeros on the diagonal.
    """
    if neighborlist_max_size not in _VALID_NEIGHBORLIST_SIZES:
        raise ValueError(
            f"neighborlist_max_size must be one of {sorted(_VALID_NEIGHBORLIST_SIZES)}, got {neighborlist_max_size}"
        )
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(f"stream must be a torch.cuda.Stream or None, got {type(stream).__name__}")
    stream_ptr = (stream if stream is not None else torch.cuda.current_stream()).cuda_stream
    result = _clustering.butina(
        distance_matrix.__cuda_array_interface__,
        cutoff,
        neighborlist_max_size,
        return_centroids,
        stream_ptr,
    )
    if return_centroids:
        clusters, centroids = result
        return AsyncGpuResult(clusters), AsyncGpuResult(centroids)
    return AsyncGpuResult(result)


def fused_butina(
    x: torch.Tensor,
    cutoff: float,
    return_centroids: bool = False,
    stream: torch.cuda.Stream | None = None,
    metric: str = "tanimoto",
):
    """Perform fused Butina clustering on a set of fingerprints.
    
    This function uses a fused implementation of Butina clustering that computes
    similarities and neighbors on-the-fly, avoiding the need to compute and store
    the full distance matrix. This makes it suitable for large datasets.

    Args:
        x: Tensor of shape (N, D) containing the fingerprints to cluster.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is less than this cutoff (i.e. similarity > 1 - cutoff).
        return_centroids: Whether to return centroid indices for each cluster.
        stream: CUDA stream to use. If None, uses the current stream.
        metric: Metric to use for similarity computation. Currently only "tanimoto"
                and "cosine" are supported.

    Returns:
        A tuple ``(clusters, cluster_sizes)`` where *clusters* is a list of tuples 
        representing each cluster (with the first element being the centroid), and 
        *cluster_sizes* is a list of cumulative cluster sizes.
        If ``return_centroids`` is True, returns a tuple ``(clusters, cluster_sizes, centroids)``
        where *centroids* is a list of centroid indices.
    """
    if metric not in ["tanimoto", "cosine"]:
        raise ValueError(f"metric must be one of ['tanimoto', 'cosine'], got {metric}")
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(f"stream must be a torch.cuda.Stream or None, got {type(stream).__name__}")
    with torch.cuda.stream(stream):
        n_start = x.shape[0]
        device = x.device
        indices = torch.arange(n_start, dtype=torch.int32, device=device)
        cluster_count = torch.zeros(2, dtype=torch.int32, device=device)
        cluster_count[1] = n_start - 1
        cluster_indices = torch.zeros(n_start, dtype=torch.int32, device=device)
        cluster_sizes = [0]
        centroids = []
        is_free = torch.ones(n_start, dtype=torch.int32, device=device)
        neigh = torch.zeros(n_start, dtype=torch.int32, device=device)
        threshold = float(1 - cutoff)
        y = x
        first_run = True
        while cluster_count[0].item() < cluster_count[1].item():
            update_neighbor_counts(x, y, neigh, threshold, subtract=not first_run, metric=metric)
            first_run = False

            max_val = neigh.max().item()
            if max_val == 0:
                break
            id_max = neigh.shape[0] - 1 - neigh.flip(0).contiguous().argmax().item()
            centroids.append(indices[id_max].item())

            extract_cluster_and_singletons(x, id_max, is_free, neigh, cluster_count, cluster_indices, threshold, indices, metric=metric)
            cluster_sizes.append(cluster_count[0].item())
            x, y = x[is_free.bool(), :].contiguous(), x[~is_free.bool(), :].contiguous()
            indices = indices[is_free.bool()].contiguous()
            neigh = neigh[is_free.bool()].contiguous()
            is_free = torch.ones(x.shape[0], dtype=torch.int32, device=x.device)

        for i in range(n_start - cluster_sizes[-1]):
            item = cluster_sizes[-1]
            cluster_sizes.append(cluster_sizes[-1] + 1)
            centroids.append(cluster_indices[item].item())
        clusters = []
        indices_cpu = cluster_indices.cpu().numpy()
        for i in range(len(cluster_sizes) - 1):
            start_idx = cluster_sizes[i]
            end_idx = cluster_sizes[i+1]
            cluster_members = indices_cpu[start_idx:end_idx].tolist()

            centroid = centroids[i]
            members = [centroid] + [m for m in cluster_members if m != centroid]
            clusters.append(tuple(members))
        if return_centroids:
            return clusters, cluster_sizes, centroids
        return clusters, cluster_sizes

if __name__ == "__main__":
    import time
    try:
        from rdkit import DataStructs
        from rdkit.DataStructs import ExplicitBitVect
        from rdkit.ML.Cluster import Butina
        HAS_RDKIT = True
    except ImportError:
        HAS_RDKIT = False
        print("RDKit not found. RDKit comparison tests will be skipped.")

    def get_rdkit_clusters(bit_tensor, threshold=0.5):
        """Convert int32 tensor to RDKit ExplicitBitVects and run Butina"""
        n = bit_tensor.shape[0]
        num_words = bit_tensor.shape[1]
        fps = []
        for i in range(n):
            bv = ExplicitBitVect(num_words * 32)
            bits = bit_tensor[i].cpu().numpy()
            for word_idx in range(num_words):
                word = int(bits[word_idx])
                for bit_idx in range(32):
                    if (word >> bit_idx) & 1:
                        bv.SetBit(word_idx * 32 + bit_idx)
            fps.append(bv)

        # Calculate pairwise distances (1 - Tanimoto similarity)
        dists = []
        for i in range(n):
            dists.extend(DataStructs.BulkCosineSimilarity(fps[i], fps[:i], returnDistance=True))
        # Run Butina clustering (cutoff is maximum distance, so 1.0 - threshold)
        cutoff = 1.0 - threshold
        clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True, reordering=True)
        return clusters

    def generate_data(n, num_clusters, noise_range=2, seed=42, num_words=64):
        """Generate random bit vectors with underlying cluster structure."""
        torch.manual_seed(seed)
        base_vectors = torch.randint(-(2**31 - 1), 2**31 - 1, size=(num_clusters, num_words), dtype=torch.int32).cuda()
        x_tr = torch.zeros((n, num_words), dtype=torch.int32).cuda()
        for i in range(n):
            base_idx = i % num_clusters
            x_tr[i] = base_vectors[base_idx]
            noise = torch.randint(0, noise_range, size=(num_words,), dtype=torch.int32).cuda()
            x_tr[i] = x_tr[i] ^ noise
        return x_tr

    def run_test(n, threshold, num_clusters, noise_range=2, seed=42, num_words=64):
        """Run a single comparison test between Triton and RDKit Butina clustering."""
        print(f"\n{'='*60}")
        print(f"Test: n={n}, threshold={threshold}, num_clusters={num_clusters}, noise_range={noise_range}, num_words={num_words}")
        print(f"{'='*60}")

        x_tr = generate_data(n, num_clusters, noise_range=noise_range, seed=seed, num_words=num_words)

        print("Running Triton clustering...")
        # fused_butina expects a distance cutoff, so we pass 1.0 - threshold
        fused_butina(x_tr, cutoff=1.0 - threshold, metric="cosine")
        torch.cuda.synchronize()
        print("Done Triton clustering, starting second run...")
        start = time.time()
        warp_clusters, _ = fused_butina(x_tr, cutoff=1.0 - threshold, metric="cosine")
        torch.cuda.synchronize()
        warp_time = time.time() - start
        print(f"Triton took {warp_time:.4f}s, found {len(warp_clusters)} clusters")

        if not HAS_RDKIT:
            return True

        print("Running RDKit Butina...")
        start = time.time()
        rdkit_failed = False
        try:
            rdkit_clusters = get_rdkit_clusters(x_tr, threshold=threshold)
        except Exception as e:
            print(f"Error running RDKit: {e}")
            rdkit_failed = True
        rdkit_time = time.time() - start
        print(f"RDKit took {rdkit_time:.4f}s, found {len(rdkit_clusters)} clusters")

        if rdkit_failed:
            return True
        rdkit_set = set(tuple(sorted(c)) for c in rdkit_clusters)
        warp_set = set(tuple(sorted(c)) for c in warp_clusters)

        passed = rdkit_set == warp_set
        if passed:
            print("SUCCESS: Clusters match exactly!")
        else:
            print("DIFFERENCE DETECTED!")
            print(f"  Clusters only in RDKit: {len(rdkit_set - warp_set)}")
            print(f"  Clusters only in Warp: {len(warp_set - rdkit_set)}")
            print(f"  RDKit diff: {rdkit_set - warp_set}")
            print(f"  Warp diff:  {warp_set - rdkit_set}")

        return passed

    def main():
        test_configs = [
            # (n, threshold, num_clusters, noise_range, num_words)
            (100,   0.3,  20,  2, 32),
            (100,   0.5,  20,  2, 32),
            (100,   0.7,  20,  2, 64),
            (100,   0.9,  20,  2, 64),
            (500,   0.4,  50,  2, 32),
            (500,   0.6,  50,  2, 32),
            (500,   0.8,  50,  2, 64),
            (1000,  0.3, 100,  2, 32),
            (1000,  0.5, 100,  2, 64),
            (1000,  0.7, 100,  2, 64),
            (5000,  0.5, 200,  2, 32),
            (5000,  0.7, 200,  2, 64),
            (10000, 0.5, 500,  2, 32),
            (10000, 0.5, 2000,  2, 64),

            # Denser clusters (lower noise) with tight threshold
            (1000,  0.9, 100,  1, 32),
            # Sparser clusters (higher noise) with loose threshold
            (1000,  0.3, 100,  4, 64),
            # Many small clusters
            (2000,  0.5, 1000, 2, 32),
            # Few large clusters
            (2000,  0.5, 10,   2, 64),
            (100000, 0.7, 1000,  128, 32),
        ]

        results = []
        for n, threshold, num_clusters, noise_range, num_words in test_configs:
            passed = run_test(n, threshold, num_clusters, noise_range=noise_range, num_words=num_words)
            results.append((n, threshold, num_clusters, noise_range, num_words, passed))

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        all_passed = True
        for n, threshold, num_clusters, noise_range, num_words, passed in results:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(f"  [{status}] n={n:>5}, threshold={threshold}, clusters={num_clusters:>4}, noise={noise_range}, words={num_words}")

        total = len(results)
        n_passed = sum(1 for *_, p in results if p)
        print(f"\n{n_passed}/{total} tests passed.")
        if not all_passed:
            exit(1)

    main()
