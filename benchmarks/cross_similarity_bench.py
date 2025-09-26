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

import argparse
import multiprocessing

import pandas as pd
import pyperf
import torch
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.similarity import (
    crossCosineSimilarity,
    crossTanimotoSimilarity,
    crossCosineSimilarityMemoryConstrained,
    crossTanimotoSimilarityMemoryConstrained,
)
from nvmolkit.fingerprints import MorganFingerprintGenerator

df = pd.read_csv("data/benchmark_smiles.csv")
smis = df.iloc[:, 0].to_list()[:2000]
mols = [MolFromSmiles(smi) for smi in smis]


runner = pyperf.Runner(min_time=0.01, values=3, processes=1, loops=3)
runner.metadata['description'] = f"Cross Similarity benchmark"

def rdkit_sim(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        sim = [BulkTanimotoSimilarity(fps[i], fps) for i in range(len(fps))]
    elif sim_type.lower() == "cosine":
        sim = [BulkCosineSimilarity(fps[i], fps) for i in range(len(fps))]

def _internal_mp(fps, idx, sim_type):
    if sim_type.lower() == "tanimoto":
        return BulkTanimotoSimilarity(fps[idx], fps)
    elif sim_type.lower() == "cosine":
        return BulkCosineSimilarity(fps[idx], fps)


def rdkit_sim_mp(fps, sim_type):
    with multiprocessing.Pool(16) as pool:
        sim = pool.starmap(_internal_mp, ([fps, i, sim_type] for i in range(len(fps))))


def nvmolkit_sim_gpu_only(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        sim = crossTanimotoSimilarity(fps)
    elif sim_type.lower() == "cosine":
        sim = crossCosineSimilarity(fps)
    torch.cuda.synchronize()


def nvmolkit_sim_cpu_collect(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        out = crossTanimotoSimilarityMemoryConstrained(fps)
    elif sim_type.lower() == "cosine":
        out = crossCosineSimilarityMemoryConstrained(fps)

for sim_type in ("tanimoto", "cosine"):
    for molNum in (100, 1000, 2000,):
        for fpsize in (1024,):

            generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fpsize)
            fps = [generator.GetFingerprint(mol) for mol in mols[:molNum]]
            while len(fps) < molNum:
                fps += fps
            fps = fps[:molNum]

            name = f"rdkit_{sim_type}sim_fpsize_{fpsize}_{molNum}mols"
            runner.bench_func(name, rdkit_sim, fps, sim_type, metadata={"name": name})

            name2 = f"rdkit_multiprocess_{sim_type}sim_fpsize_{fpsize}_{molNum}mols"
            runner.bench_func(name2, rdkit_sim_mp,  fps, sim_type, metadata={"name": str(name2)})

            while len(mols) < molNum:
                mols += mols
            mols = mols[:molNum]
            nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=fpsize)

            nvmolkit_fps_cu = torch.as_tensor(nvmolkit_fpgen.GetFingerprints(mols[:molNum]), device='cuda')
            name3 = f"nvmolkit_gpu-only_{sim_type}sim_fpsize_{fpsize}_{molNum}mols_gpu_only"
            runner.bench_func(name3, nvmolkit_sim_gpu_only, nvmolkit_fps_cu, sim_type, metadata={"name": str(name3)})

            name4 = f"nvmolkit_cpu-collect_{sim_type}sim_fpsize_{fpsize}_{molNum}mols_cpu_result"
            runner.bench_func(name4, nvmolkit_sim_cpu_collect, nvmolkit_fps_cu, sim_type, metadata={"name": str(name4)})
