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

"""TFD profiling script for use with NVIDIA Nsight Systems (nsys).

Designed to produce NVTX-annotated traces for analyzing CPU vs GPU TFD performance.
C++ NVTX ranges (buildTFDSystem, transferToDevice, kernel launches, etc.) are
automatically captured by nsys alongside the Python-level annotations below.

Usage:
    # Profile GPU path
    nsys profile -t nvtx,osrt,cuda --gpu-metrics-devices=all --stats=true \
        -f true -o tfd_gpu_profile \
        python benchmarks/tfd_profile.py 50 20 gpu

    # Profile CPU path
    nsys profile -t nvtx,osrt,cuda --stats=true \
        -f true -o tfd_cpu_profile \
        python benchmarks/tfd_profile.py 50 20 cpu

    # Profile both paths
    nsys profile -t nvtx,osrt,cuda --gpu-metrics-devices=all --stats=true \
        -f true -o tfd_both_profile \
        python benchmarks/tfd_profile.py 50 20 both
"""

import argparse
import sys

import nvtx
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import nvmolkit.tfd as nvmol_tfd


def generate_conformers(mol, num_confs, seed=42):
    """Generate conformers for a molecule using ETKDG."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.useRandomCoords = True
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(conf_ids) < 2:
        return None
    return Chem.RemoveHs(mol)


def prepare_molecules(smiles_file, num_mols, num_confs):
    """Load SMILES and prepare molecules with conformers."""
    import pandas as pd

    df = pd.read_csv(smiles_file)
    smiles_list = df.iloc[:, 0].tolist()

    mols = []
    for i, smi in enumerate(smiles_list):
        if len(mols) >= num_mols:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 4:
            continue
        mol_with_confs = generate_conformers(mol, num_confs, seed=42 + i)
        if mol_with_confs is not None and mol_with_confs.GetNumConformers() >= 2:
            mols.append(mol_with_confs)

    return mols


def main():
    parser = argparse.ArgumentParser(
        description="TFD profiling script for nsys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Profile GPU with 50 molecules, 20 conformers each:
  nsys profile -t nvtx,osrt,cuda --gpu-metrics-devices=all --stats=true \\
      -f true -o tfd_gpu python benchmarks/tfd_profile.py 50 20 gpu

  # Profile CPU:
  nsys profile -t nvtx,osrt,cuda --stats=true \\
      -f true -o tfd_cpu python benchmarks/tfd_profile.py 50 20 cpu
""",
    )
    parser.add_argument("num_mols", type=int, help="Number of molecules")
    parser.add_argument("num_confs", type=int, help="Number of conformers per molecule")
    parser.add_argument(
        "backend",
        choices=["cpu", "gpu", "gpu_buffer", "both"],
        help="Backend to profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of profiled runs (default: 3)",
    )
    parser.add_argument(
        "--smiles-file",
        type=str,
        default="data/benchmark_smiles.csv",
        help="CSV file with SMILES (default: data/benchmark_smiles.csv)",
    )
    args = parser.parse_args()

    # === Setup ===
    with nvtx.annotate("Setup", color="blue"):
        print(f"Loading molecules: {args.num_mols} mols x {args.num_confs} confs")
        mols = prepare_molecules(args.smiles_file, args.num_mols, args.num_confs)
        if not mols:
            print("Error: No valid molecules prepared")
            sys.exit(1)

        actual_confs = [mol.GetNumConformers() for mol in mols]
        total_pairs = sum(c * (c - 1) // 2 for c in actual_confs)
        print(f"Prepared {len(mols)} molecules, avg {sum(actual_confs)/len(actual_confs):.1f} conformers")
        print(f"Total TFD pairs: {total_pairs}")
        print(f"Backend: {args.backend}")

    # === Warmup ===
    if args.warmup > 0:
        with nvtx.annotate("Warmup", color="red"):
            print(f"\nWarmup ({args.warmup} iteration(s))...")
            warmup_mols = mols[:min(5, len(mols))]
            for i in range(args.warmup):
                if args.backend in ("cpu", "both"):
                    nvmol_tfd.GetTFDMatrices(warmup_mols, useWeights=True, maxDev="equal", backend="cpu")
                if args.backend in ("gpu", "gpu_buffer", "both"):
                    nvmol_tfd.GetTFDMatrices(warmup_mols, useWeights=True, maxDev="equal", backend="gpu")
                    torch.cuda.synchronize()

    # === Profiled runs ===
    with nvtx.annotate("Profiled runs", color="green"):
        print(f"\nRunning {args.runs} profiled iteration(s)...")

        for run in range(args.runs):
            if args.backend in ("cpu", "both"):
                with nvtx.annotate(f"CPU run {run}", color="cyan"):
                    results = nvmol_tfd.GetTFDMatrices(
                        mols, useWeights=True, maxDev="equal", backend="cpu"
                    )
                    total_values = sum(len(r) for r in results)
                    print(f"  CPU run {run}: {total_values} TFD values computed")

            if args.backend in ("gpu", "both"):
                with nvtx.annotate(f"GPU run {run}", color="orange"):
                    results = nvmol_tfd.GetTFDMatrices(
                        mols, useWeights=True, maxDev="equal", backend="gpu"
                    )
                    torch.cuda.synchronize()
                    total_values = sum(len(r) for r in results)
                    print(f"  GPU run {run}: {total_values} TFD values computed")

            if args.backend in ("gpu_buffer", "both"):
                with nvtx.annotate(f"GPU buffer run {run}", color="yellow"):
                    result = nvmol_tfd.GetTFDMatricesGpu(
                        mols, useWeights=True, maxDev="equal"
                    )
                    torch.cuda.synchronize()
                    print(f"  GPU buffer run {run}: done")

    print("\nProfiling complete.")


if __name__ == "__main__":
    main()
