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

import pandas as pd
import pyperf
import torch
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.similarity import crossCosineSimilarity, crossTanimotoSimilarity
from nvmolkit.fingerprints import MorganFingerprintGenerator


SIZES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 24000, 28000, 32000]


def rdkit_sim(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        [BulkTanimotoSimilarity(fps[i], fps) for i in range(len(fps))]
    elif sim_type.lower() == "cosine":
        [BulkCosineSimilarity(fps[i], fps) for i in range(len(fps))]


def nvmolkit_sim_gpu_only(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        crossTanimotoSimilarity(fps)
    elif sim_type.lower() == "cosine":
        crossCosineSimilarity(fps)
    torch.cuda.synchronize()


runner = pyperf.Runner(min_time=0.01, values=3, processes=1, loops=3)
runner.metadata["description"] = "Cross Similarity benchmark"
runner.argparser.add_argument(
    "--input", type=str, default="data/benchmark_smiles.csv", help="Path to input SMILES CSV file"
)
runner.argparser.add_argument("--cosine", action="store_true", help="Include cosine similarity benchmarks")
args = runner.parse_args()

sim_types = ("tanimoto", "cosine") if args.cosine else ("tanimoto",)
fpsize = 1024
max_size = max(SIZES)

df = pd.read_csv(args.input)
smis = df.iloc[:, 0].to_list()
mols = [MolFromSmiles(smi) for smi in smis]
mols = [mol for mol in mols if mol is not None]
if not mols:
    raise ValueError(f"No molecules parsed from {args.input}")
while len(mols) < max_size:
    mols += mols
mols = mols[:max_size]

rdkit_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fpsize)
rdkit_fps_all = [rdkit_fpgen.GetFingerprint(mol) for mol in mols]
nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=fpsize)
nvmolkit_fps_all = torch.as_tensor(nvmolkit_fpgen.GetFingerprints(mols), device="cuda")

for sim_type in sim_types:
    for molNum in SIZES:
        fps = rdkit_fps_all[:molNum]
        nvmolkit_fps_cu = nvmolkit_fps_all[:molNum].contiguous()

        name = f"rdkit_{sim_type}sim_fpsize_{fpsize}_{molNum}mols"
        runner.bench_func(name, rdkit_sim, fps, sim_type, metadata={"name": name})

        name2 = f"nvmolkit_gpu-only_{sim_type}sim_fpsize_{fpsize}_{molNum}mols_gpu_only"
        runner.bench_func(name2, nvmolkit_sim_gpu_only, nvmolkit_fps_cu, sim_type, metadata={"name": name2})
