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

import multiprocessing

import pandas as pd
import pyperf
import torch
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.similarity import bulkCosineSimilarity as nvMolKitCosineSimilarity
from nvmolkit.similarity import bulkTanimotoSimilarity as nvMolKitTanimotoSimilarity
from nvmolkit.fingerprints import MorganFingerprintGenerator

df = pd.read_csv("data/benchmark_smiles.csv")
smis = df.iloc[:, 0].to_list()[:2000]
mols = [MolFromSmiles(smi) for smi in smis]

runner = pyperf.Runner(min_time=0.01, values=3, processes=1, loops=3)
runner.metadata['description'] = "Similarity benchmark"

def rdkit_sim(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        sim = BulkTanimotoSimilarity(fps[0], fps)
    elif sim_type.lower() == "cosine":
        sim = BulkCosineSimilarity(fps[0], fps)


def _internal_mp(fp0, fps, sim_type):
    if sim_type.lower() == "tanimoto":
        return BulkTanimotoSimilarity(fp0, fps)
    elif sim_type.lower() == "cosine":
        return BulkCosineSimilarity(fp0, fps)


def rdkit_sim_mp(fps, chunksize, sim_type):
    chunks = [fps[i:i + chunksize] for i in range(0, len(fps), chunksize)]
    with multiprocessing.Pool(16) as pool:
        sim = pool.starmap(_internal_mp, [(fps[0], chunk, sim_type) for chunk in chunks])


def nvmolkit_sim(fps, backend, sim_type):
    if sim_type.lower() == "tanimoto":
        sim = nvMolKitTanimotoSimilarity(fps[0], fps, backend=backend)
    elif sim_type.lower() == "cosine":
        sim = nvMolKitCosineSimilarity(fps[0], fps, backend=backend)


def nvmolkit_sim_gpu_only(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        sim = torch.as_tensor(nvMolKitTanimotoSimilarity(fps[0, :], fps), device='cuda')
    elif sim_type.lower() == "cosine":
        sim = torch.as_tensor(nvMolKitCosineSimilarity(fps[0, :], fps), device='cuda')
    torch.cuda.synchronize()


for sim_type in ("tanimoto", "cosine"):
    for molNum in (2000,):
        for fpsize in (128, 512, 1024, 2048):

            generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fpsize)
            fps = [generator.GetFingerprint(mol) for mol in mols[:molNum]]
            while len(fps) < molNum:
                fps += fps
            fps = fps[:molNum]

            name = f"rdkit_{sim_type}sim_fpsize_{fpsize}_{molNum}mols"
            runner.bench_func(name, rdkit_sim, fps, sim_type, metadata={"name": name})

            for chunksize in (1000,):
                name2 = f"rdkit_multiprocess_{sim_type}sim_fpsize_{fpsize}_{molNum}mols_chunksize_{chunksize}"
                runner.bench_func(name2, rdkit_sim_mp,  fps,  chunksize, sim_type, metadata={"name": str(name2)})

            name3 = f"nvmolkit_cpu_{sim_type}sim_fpsize_{fpsize}_{molNum}mols"
            runner.bench_func(name3, nvmolkit_sim,  fps, 'cpu', sim_type, metadata={"name": str(name3)})

            name4 = f"nvmolkit_gpu_{sim_type}sim_fpsize_{fpsize}_{molNum}mols"
            runner.bench_func(name4, nvmolkit_sim,  fps, 'gpu', sim_type, metadata={"name": str(name4)})

            nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=fpsize)
            nvmolkit_fps_cu = torch.as_tensor(nvmolkit_fpgen.GetFingerprints(mols[:molNum]), device='cuda')
            name5 = f"nvmolkit_gpu-only_{sim_type}sim_fpsize_{fpsize}_{molNum}mols_gpu_only"
            runner.bench_func(name5, nvmolkit_sim_gpu_only, nvmolkit_fps_cu, sim_type, metadata={"name": str(name5)})
