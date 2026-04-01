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

import sys

import pandas as pd
import torch
from benchmark_timing import time_it

from nvmolkit.clustering import fused_butina

try:
    from rdkit import DataStructs
    from rdkit.DataStructs import ExplicitBitVect
    from rdkit.ML.Cluster import Butina

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("RDKit not found. RDKit comparison will be skipped.")


def generate_data(n, num_clusters, noise_range=2, seed=42, num_words=64):
    """Generate random bit vectors with underlying cluster structure."""
    torch.manual_seed(seed)
    base_vectors = torch.randint(
        -(2**31 - 1), 2**31 - 1, size=(num_clusters, num_words), dtype=torch.int32, device="cuda"
    )
    x = torch.zeros((n, num_words), dtype=torch.int32, device="cuda")
    for i in range(n):
        x[i] = base_vectors[i % num_clusters]
        noise = torch.randint(0, noise_range, size=(num_words,), dtype=torch.int32, device="cuda")
        x[i] = x[i] ^ noise
    return x


def get_rdkit_clusters(bit_tensor, threshold=0.5, metric="tanimoto"):
    """Convert int32 tensor to RDKit ExplicitBitVects and run Butina."""
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

    bulk_sim_fn = DataStructs.BulkTanimotoSimilarity if metric == "tanimoto" else DataStructs.BulkCosineSimilarity
    dists = []
    for i in range(n):
        dists.extend(bulk_sim_fn(fps[i], fps[:i], returnDistance=True))
    cutoff = 1.0 - threshold
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True, reordering=True)
    return clusters


def run_test(n, threshold, num_clusters, noise_range=2, seed=42, num_words=64, metric="tanimoto"):
    """Run a single comparison test between fused Butina and RDKit."""
    print(f"\n{'=' * 60}")
    print(
        f"Test: n={n}, threshold={threshold}, clusters={num_clusters}, noise={noise_range}, words={num_words}, metric={metric}"
    )
    print(f"{'=' * 60}")

    x = generate_data(n, num_clusters, noise_range=noise_range, seed=seed, num_words=num_words)
    cutoff = 1.0 - threshold

    triton_result = time_it(lambda: fused_butina(x, cutoff=cutoff, metric=metric), gpu_sync=True)
    warp_clusters, _ = fused_butina(x, cutoff=cutoff, metric=metric)
    torch.cuda.synchronize()
    print(f"Triton: {triton_result.median_ms:.2f}ms (median), found {len(warp_clusters)} clusters")

    rdkit_time_ms = 0.0
    passed = True
    if HAS_RDKIT:
        try:
            rdkit_result = time_it(lambda: get_rdkit_clusters(x, threshold=threshold, metric=metric), runs=1)
            rdkit_clusters = get_rdkit_clusters(x, threshold=threshold, metric=metric)
            rdkit_time_ms = rdkit_result.median_ms
            print(f"RDKit:  {rdkit_time_ms:.2f}ms (median), found {len(rdkit_clusters)} clusters")

            rdkit_set = {tuple(sorted(c)) for c in rdkit_clusters}
            warp_set = {tuple(sorted(c)) for c in warp_clusters}
            passed = rdkit_set == warp_set
            if passed:
                print("SUCCESS: Clusters match exactly!")
            else:
                print("DIFFERENCE DETECTED!")
                print(f"  Clusters only in RDKit: {len(rdkit_set - warp_set)}")
                print(f"  Clusters only in Triton: {len(warp_set - rdkit_set)}")
        except Exception as e:
            print(f"Error running RDKit: {e}")

    return {
        "n": n,
        "threshold": threshold,
        "num_clusters": num_clusters,
        "noise_range": noise_range,
        "num_words": num_words,
        "metric": metric,
        "triton_median_ms": triton_result.median_ms,
        "triton_std_ms": triton_result.std_ms,
        "rdkit_median_ms": rdkit_time_ms,
        "passed": passed,
    }


if __name__ == "__main__":
    metric = sys.argv[1] if len(sys.argv) > 1 else "tanimoto"
    if metric not in ("tanimoto", "cosine"):
        print("Usage: python fused_butina_clustering_bench.py [tanimoto|cosine]")
        sys.exit(1)

    test_configs = [
        # (n, threshold, num_clusters, noise_range, num_words)
        (100, 0.3, 20, 2, 32),
        (100, 0.5, 20, 2, 32),
        (100, 0.7, 20, 2, 64),
        (100, 0.9, 20, 2, 64),
        (500, 0.4, 50, 2, 32),
        (500, 0.6, 50, 2, 32),
        (500, 0.8, 50, 2, 64),
        (1000, 0.3, 100, 2, 32),
        (1000, 0.5, 100, 2, 64),
        (1000, 0.7, 100, 2, 64),
        (5000, 0.5, 200, 2, 32),
        (5000, 0.7, 200, 2, 64),
        (10000, 0.5, 500, 2, 32),
        (10000, 0.5, 2000, 2, 64),
        # Denser clusters (lower noise) with tight threshold
        (1000, 0.9, 100, 1, 32),
        # Sparser clusters (higher noise) with loose threshold
        (1000, 0.3, 100, 4, 64),
        # Many small clusters
        (2000, 0.5, 1000, 2, 32),
        # Few large clusters
        (2000, 0.5, 10, 2, 64),
        # (100000, 0.7, 1000, 128, 32),
    ]

    results = []
    try:
        for n, threshold, num_clusters, noise_range, num_words in test_configs:
            result = run_test(n, threshold, num_clusters, noise_range=noise_range, num_words=num_words, metric=metric)
            results.append(result)
    except Exception as e:
        print(f"Got exception: {e}, exiting early")

    df = pd.DataFrame(results)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))

    all_passed = all(r["passed"] for r in results)
    n_passed = sum(1 for r in results if r["passed"])
    print(f"\n{n_passed}/{len(results)} tests passed.")

    df.to_csv("fused_butina_results.csv", index=False)
    if not all_passed:
        sys.exit(1)
