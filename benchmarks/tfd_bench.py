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

"""Benchmark for TFD (Torsion Fingerprint Deviation) calculation.

Compares:
- RDKit TorsionFingerprints.GetTFDMatrix (Python, single-threaded)
- nvMolKit CPU backend (C++ with OpenMP, multi-threaded)
- nvMolKit GPU backend (CUDA)

Usage:
    python tfd_bench.py [--smiles-file FILE] [--output FILE] [--skip-rdkit]

Example:
    python tfd_bench.py --smiles-file data/benchmark_smiles.csv --output tfd_results.csv
"""

import argparse
import sys
import time
from typing import List, Tuple

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints

import nvmolkit.tfd as nvmol_tfd


def time_it(func, runs: int = 3, warmups: int = 1) -> Tuple[float, float]:
    """Time a function with warmup runs.

    Args:
        func: Function to time (no arguments)
        runs: Number of timed runs
        warmups: Number of warmup runs

    Returns:
        Tuple of (average_time_ms, std_time_ms)
    """
    for _ in range(warmups):
        func()

    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)

    avg_time = sum(times) / runs
    std_time = (sum((t - avg_time) ** 2 for t in times) / runs) ** 0.5
    return avg_time / 1.0e6, std_time / 1.0e6  # Return in milliseconds


def generate_conformers(mol: Chem.Mol, num_confs: int, seed: int = 42) -> Chem.Mol:
    """Generate conformers for a molecule using ETKDG.

    Args:
        mol: RDKit molecule
        num_confs: Number of conformers to generate
        seed: Random seed

    Returns:
        Molecule with conformers (or None if embedding failed)
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1  # Single-threaded for reproducibility
    params.useRandomCoords = True

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    if len(conf_ids) < 2:
        return None

    mol = Chem.RemoveHs(mol)
    return mol


def prepare_molecules(smiles_list: List[str], num_confs: int, max_mols: int = 100) -> List[Chem.Mol]:
    """Prepare molecules with conformers for benchmarking.

    Args:
        smiles_list: List of SMILES strings
        num_confs: Number of conformers per molecule
        max_mols: Maximum number of molecules to prepare

    Returns:
        List of molecules with conformers
    """
    mols = []
    for i, smi in enumerate(smiles_list):
        if len(mols) >= max_mols:
            break

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Skip very small molecules (no torsions)
        if mol.GetNumAtoms() < 4:
            continue

        mol_with_confs = generate_conformers(mol, num_confs, seed=42 + i)
        if mol_with_confs is not None and mol_with_confs.GetNumConformers() >= 2:
            mols.append(mol_with_confs)

    return mols


def bench_rdkit_single(mol: Chem.Mol) -> None:
    """Benchmark RDKit TFD for a single molecule."""
    TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")


def bench_rdkit_batch(mols: List[Chem.Mol]) -> None:
    """Benchmark RDKit TFD for multiple molecules (sequential)."""
    for mol in mols:
        TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")


def bench_nvmol_cpu_single(mol: Chem.Mol) -> None:
    """Benchmark nvMolKit CPU TFD for a single molecule."""
    nvmol_tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal", backend="cpu")


def bench_nvmol_cpu_batch(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit CPU TFD for multiple molecules."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", backend="cpu")


def bench_nvmol_gpu_single(mol: Chem.Mol) -> None:
    """Benchmark nvMolKit GPU TFD for a single molecule."""
    nvmol_tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal", backend="gpu")
    torch.cuda.synchronize()


def bench_nvmol_gpu_batch(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD for multiple molecules."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", backend="gpu")
    torch.cuda.synchronize()


def bench_nvmol_gpu_buffer(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD with GPU-resident output."""
    result = nvmol_tfd.GetTFDMatricesGpu(mols, useWeights=True, maxDev="equal")
    torch.cuda.synchronize()


def verify_correctness(mol: Chem.Mol, tolerance: float = 0.01) -> bool:
    """Verify nvMolKit results match RDKit (within tolerance).

    Multi-quartet torsions (rings and symmetric) are fully supported,
    so results should match RDKit closely.
    """
    rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
    nvmol_result = nvmol_tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal", backend="gpu")

    if len(rdkit_result) != len(nvmol_result):
        return False

    for rd, nv in zip(rdkit_result, nvmol_result):
        if abs(rd - nv) > tolerance:
            return False

    return True


def run_benchmarks(
    smiles_list: List[str],
    skip_rdkit: bool = False,
    output_file: str = "tfd_results.csv",
) -> pd.DataFrame:
    """Run TFD benchmarks with various configurations.

    Args:
        smiles_list: List of SMILES strings
        skip_rdkit: If True, skip RDKit benchmarks (faster for large runs)
        output_file: Output CSV file path

    Returns:
        DataFrame with benchmark results
    """
    results = []

    # Test configurations
    mol_counts = [1, 5, 10, 25, 50, 100]
    conformer_counts = [5, 10, 20]

    print("=" * 70)
    print("TFD Benchmark: RDKit vs nvMolKit (CPU) vs nvMolKit (GPU)")
    print("=" * 70)

    for num_confs in conformer_counts:
        print(f"\n--- Preparing molecules with {num_confs} conformers ---")

        # Prepare molecules (use larger pool for selection)
        all_mols = prepare_molecules(smiles_list, num_confs, max_mols=max(mol_counts) + 20)

        if len(all_mols) < max(mol_counts):
            print(f"Warning: Only {len(all_mols)} molecules available")

        for num_mols in mol_counts:
            if num_mols > len(all_mols):
                print(f"Skipping {num_mols} mols (only {len(all_mols)} available)")
                continue

            mols = all_mols[:num_mols]
            actual_confs = [mol.GetNumConformers() for mol in mols]
            avg_confs = sum(actual_confs) / len(actual_confs)

            print(f"\nBenchmarking: {num_mols} molecules, ~{avg_confs:.1f} conformers each")

            # Calculate expected TFD pairs
            total_pairs = sum(c * (c - 1) // 2 for c in actual_confs)
            print(f"  Total TFD pairs: {total_pairs}")

            result = {
                "num_molecules": num_mols,
                "target_conformers": num_confs,
                "avg_conformers": avg_confs,
                "total_tfd_pairs": total_pairs,
            }

            # RDKit benchmark (single-threaded Python)
            if not skip_rdkit:
                try:
                    rdkit_time, rdkit_std = time_it(lambda: bench_rdkit_batch(mols))
                    result["rdkit_time_ms"] = rdkit_time
                    result["rdkit_std_ms"] = rdkit_std
                    print(f"  RDKit (Python):     {rdkit_time:8.2f} ms (+/- {rdkit_std:.2f})")
                except Exception as e:
                    print(f"  RDKit failed: {e}")
                    result["rdkit_time_ms"] = None
                    result["rdkit_std_ms"] = None
            else:
                result["rdkit_time_ms"] = None
                result["rdkit_std_ms"] = None

            # nvMolKit CPU benchmark (OpenMP multi-threaded)
            try:
                nvmol_cpu_time, nvmol_cpu_std = time_it(lambda: bench_nvmol_cpu_batch(mols))
                result["nvmol_cpu_time_ms"] = nvmol_cpu_time
                result["nvmol_cpu_std_ms"] = nvmol_cpu_std
                print(f"  nvMolKit (CPU):     {nvmol_cpu_time:8.2f} ms (+/- {nvmol_cpu_std:.2f})")
            except Exception as e:
                print(f"  nvMolKit CPU failed: {e}")
                result["nvmol_cpu_time_ms"] = None
                result["nvmol_cpu_std_ms"] = None

            # nvMolKit GPU benchmark
            try:
                nvmol_gpu_time, nvmol_gpu_std = time_it(lambda: bench_nvmol_gpu_batch(mols))
                result["nvmol_gpu_time_ms"] = nvmol_gpu_time
                result["nvmol_gpu_std_ms"] = nvmol_gpu_std
                print(f"  nvMolKit (GPU):     {nvmol_gpu_time:8.2f} ms (+/- {nvmol_gpu_std:.2f})")
            except Exception as e:
                print(f"  nvMolKit GPU failed: {e}")
                result["nvmol_gpu_time_ms"] = None
                result["nvmol_gpu_std_ms"] = None

            # nvMolKit GPU-resident benchmark
            try:
                nvmol_gpu_buf_time, nvmol_gpu_buf_std = time_it(lambda: bench_nvmol_gpu_buffer(mols))
                result["nvmol_gpu_buffer_time_ms"] = nvmol_gpu_buf_time
                result["nvmol_gpu_buffer_std_ms"] = nvmol_gpu_buf_std
                print(f"  nvMolKit (GPU buf): {nvmol_gpu_buf_time:8.2f} ms (+/- {nvmol_gpu_buf_std:.2f})")
            except Exception as e:
                print(f"  nvMolKit GPU buffer failed: {e}")
                result["nvmol_gpu_buffer_time_ms"] = None
                result["nvmol_gpu_buffer_std_ms"] = None

            # Calculate speedups
            if result.get("rdkit_time_ms") and result.get("nvmol_cpu_time_ms"):
                result["speedup_cpu_vs_rdkit"] = result["rdkit_time_ms"] / result["nvmol_cpu_time_ms"]
            else:
                result["speedup_cpu_vs_rdkit"] = None

            if result.get("rdkit_time_ms") and result.get("nvmol_gpu_time_ms"):
                result["speedup_gpu_vs_rdkit"] = result["rdkit_time_ms"] / result["nvmol_gpu_time_ms"]
            else:
                result["speedup_gpu_vs_rdkit"] = None

            if result.get("nvmol_cpu_time_ms") and result.get("nvmol_gpu_time_ms"):
                result["speedup_gpu_vs_cpu"] = result["nvmol_cpu_time_ms"] / result["nvmol_gpu_time_ms"]
            else:
                result["speedup_gpu_vs_cpu"] = None

            # Print speedups
            if result.get("speedup_cpu_vs_rdkit"):
                print(f"  Speedup CPU vs RDKit: {result['speedup_cpu_vs_rdkit']:.1f}x")
            if result.get("speedup_gpu_vs_rdkit"):
                print(f"  Speedup GPU vs RDKit: {result['speedup_gpu_vs_rdkit']:.1f}x")
            if result.get("speedup_gpu_vs_cpu"):
                print(f"  Speedup GPU vs CPU:   {result['speedup_gpu_vs_cpu']:.1f}x")

            results.append(result)

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")

    return df


def main():
    parser = argparse.ArgumentParser(description="TFD Benchmark")
    parser.add_argument(
        "--smiles-file",
        type=str,
        default="data/benchmark_smiles.csv",
        help="CSV file with SMILES (first column)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tfd_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--skip-rdkit",
        action="store_true",
        help="Skip RDKit benchmarks (faster)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify correctness before benchmarking",
    )
    args = parser.parse_args()

    # Load SMILES
    print(f"Loading SMILES from: {args.smiles_file}")
    try:
        df = pd.read_csv(args.smiles_file)
        smiles_list = df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error loading SMILES file: {e}")
        sys.exit(1)

    print(f"Loaded {len(smiles_list)} SMILES")

    # Optional: Verify correctness
    if args.verify:
        print("\nVerifying correctness...")
        test_mols = prepare_molecules(smiles_list[:20], num_confs=5, max_mols=5)
        all_correct = True
        for i, mol in enumerate(test_mols):
            if verify_correctness(mol):
                print(f"  Molecule {i}: OK")
            else:
                print(f"  Molecule {i}: MISMATCH")
                all_correct = False
        if not all_correct:
            print("Warning: Some molecules did not match RDKit within tolerance")

    # Run benchmarks
    run_benchmarks(
        smiles_list,
        skip_rdkit=args.skip_rdkit,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
