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
from rdkit.Chem import MolFromSmiles
from rdkit.ML.Cluster.Butina import ClusterData
import torch

from nvmolkit.clustering import butina as butina_nvmol
from nvmolkit.fingerprints import MorganFingerprintGenerator as nvmolMorganGen
from nvmolkit.similarity import crossTanimotoSimilarity


def get_distance_matrix(molecules):
    nvmol_gen = nvmolMorganGen(radius=2, fpSize=1024)
    nvmol_fps = nvmol_gen.GetFingerprints(molecules, 10)
    sim_matrix = crossTanimotoSimilarity(nvmol_fps).torch()
    return 1.0 - sim_matrix


def resize_and_fill(distance_mat: torch.Tensor, want_size):
    current_size = distance_mat.shape[0]
    if current_size >= want_size:
        return distance_mat[:want_size, :want_size].contiguous()
    full_mat = torch.rand(want_size, want_size, dtype=distance_mat.dtype, device=distance_mat.device)
    full_mat = torch.abs(full_mat - full_mat.T).clip(0.01, 0.99)
    full_mat.fill_diagonal_(0.0)
    full_mat[:current_size, :current_size] = distance_mat
    return full_mat


def time_it(func, runs=3, warmups=1):
    import time

    for _ in range(warmups):
        func()
    times = []
    for _ in range(runs):
        start = time.time_ns()
        func()
        end = time.time_ns()
        times.append(end - start)
    avg_time = sum(times) / runs
    std_time = (sum((t - avg_time) ** 2 for t in times) / runs) ** 0.5
    return avg_time / 1.0e6, std_time / 1.0e6  # return in milliseconds


def bench_rdkit(data, threshold):
    return time_it(lambda: ClusterData(data, len(data), threshold, isDistData=True, reordering=True))


MAX_BENCH_SIZE = 40000

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python butina_clustering_bench.py <input_smiles_file> <do_rdkit (0 or 1)>")
        sys.exit(1)
    input_data = sys.argv[1]
    do_rdkit = sys.argv[2] != "0"

    with open(input_data, "r") as f:
        smis = [line.strip() for line in f.readlines()]
    mols = [MolFromSmiles(smi, sanitize=True) for smi in smis[:MAX_BENCH_SIZE + 100]]
    mols = [mol for mol in mols if mol is not None]

    dists = get_distance_matrix(mols)

    sizes = [1000, 5000, 10000, 20000, 30000, 40000]
    cutoffs = [1e-10, 0.1, 0.2, 1.1]
    results = []

    try:
        for size in sizes:
            for cutoff in cutoffs:
                # Don't run large sizes for edge cases.
                if cutoff in (1e-10, 1.1) and size > 20000:
                    continue
                print(f"Running size {size} cutoff {cutoff}")
                dist_mat = resize_and_fill(dists, size)
                if do_rdkit:
                    dist_mat_numpy = dist_mat.cpu().numpy()
                    rdkit_time, rdk_std = bench_rdkit(dist_mat_numpy, cutoff)
                else:
                    rdkit_time = 0.0
                    rdk_std = 0.0
                nvmol_time, nvmol_std = time_it(lambda: butina_nvmol(dist_mat, cutoff))

                results.append(
                    {
                        "size": size,
                        "cutoff": cutoff,
                        "rdkit_time_ms": rdkit_time,
                        "rdkit_std_ms": rdk_std,
                        "nvmol_time_ms": nvmol_time,
                        "nvmol_std_ms": nvmol_std,
                    }
                )
    except Exception as e:
        print(f"Got exception: {e}, exiting early")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results.csv", index=False)
