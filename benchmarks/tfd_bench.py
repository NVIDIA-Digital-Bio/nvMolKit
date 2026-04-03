# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    python tfd_bench.py --pkl-file data1.pkl data2.pkl [--output FILE]

Example:
    python tfd_bench.py --smiles-file data/benchmark_smiles.csv --output tfd_results.csv

    # Use precomputed ChEMBL stratified pickles directly:
    python tfd_bench.py --pkl-file ../Data/Chembl_stratified_prepared/chembl_0-20_10confs.pkl \
        --num-mols 100 500 1000 5000 --output tfd_chembl_results.csv
"""

import argparse
import os
import pickle
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


def _try_load_pickle(num_confs: int, max_mols: int, smiles_file: str = None) -> List[Chem.Mol]:
    """Try to load precomputed molecules from pickle file."""
    search_dirs = []
    if smiles_file:
        search_dirs.append(os.path.dirname(os.path.abspath(smiles_file)))
    search_dirs.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

    for d in search_dirs:
        pkl_path = os.path.join(d, f"prepared_mols_{num_confs}confs.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                all_mols = pickle.load(f)
            mols = all_mols[:max_mols]
            print(f"  Loaded {len(mols)} molecules from {pkl_path}")
            return mols
    return None


def prepare_molecules(
    smiles_list: List[str], num_confs: int, max_mols: int = 100, smiles_file: str = None
) -> List[Chem.Mol]:
    """Prepare molecules with conformers, using precomputed pickle if available.

    Args:
        smiles_list: List of SMILES strings (fallback if no pickle)
        num_confs: Number of conformers per molecule
        max_mols: Maximum number of molecules to prepare
        smiles_file: Path to SMILES CSV (used to locate pickle files)

    Returns:
        List of molecules with conformers
    """
    cached = _try_load_pickle(num_confs, max_mols, smiles_file)
    if cached is not None:
        return cached

    print(f"  No precomputed pickle found, generating from scratch...")
    mols = []
    for i, smi in enumerate(smiles_list):
        if len(mols) >= max_mols:
            break

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

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


def bench_nvmol_cpu_list(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit CPU TFD returning Python lists."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", backend="cpu", return_type="list")


def bench_nvmol_cpu_numpy(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit CPU TFD returning numpy arrays."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", backend="cpu", return_type="numpy")


def bench_nvmol_gpu_single(mol: Chem.Mol) -> None:
    """Benchmark nvMolKit GPU TFD for a single molecule."""
    nvmol_tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal", backend="gpu")
    torch.cuda.synchronize()


def bench_nvmol_gpu_list(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD returning Python lists."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", backend="gpu", return_type="list")
    torch.cuda.synchronize()


def bench_nvmol_gpu_numpy(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD returning numpy arrays."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", backend="gpu", return_type="numpy")
    torch.cuda.synchronize()


def bench_nvmol_gpu_tensor(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD returning GPU tensors (no D2H)."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", backend="gpu", return_type="tensor")
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


def load_pkl_files(pkl_paths: List[str]) -> List[Chem.Mol]:
    """Load and concatenate molecules from one or more pickle files.

    Each pickle file must contain a list of RDKit Mol objects with conformers
    (as produced by prepare_chembl_stratified.py).
    """
    all_mols = []
    for path in pkl_paths:
        with open(path, "rb") as f:
            mols = pickle.load(f)
        print(f"  Loaded {len(mols)} molecules from {path}")
        all_mols.extend(mols)
    return all_mols


def run_benchmarks(
    smiles_list: List[str] | None = None,
    skip_rdkit: bool = False,
    output_file: str = "tfd_results.csv",
    smiles_file: str = None,
    mol_counts: List[int] = None,
    conformer_counts: List[int] = None,
    preloaded_mols: List[Chem.Mol] | None = None,
) -> pd.DataFrame:
    """Run TFD benchmarks with various configurations.

    Args:
        smiles_list: List of SMILES strings (unused when preloaded_mols given)
        skip_rdkit: If True, skip RDKit benchmarks (faster for large runs)
        output_file: Output CSV file path
        smiles_file: Path to SMILES CSV (used to locate pickle files)
        mol_counts: List of molecule counts to benchmark
        conformer_counts: List of conformer counts to benchmark
        preloaded_mols: Pre-loaded molecules with conformers (e.g. from --pkl-file).
            When provided, smiles_list/smiles_file/conformer_counts are ignored and
            the actual conformer count is read from the molecules.

    Returns:
        DataFrame with benchmark results
    """
    if mol_counts is None:
        mol_counts = [1, 5, 10, 25, 50, 100]

    if preloaded_mols is not None:
        actual_confs_all = [m.GetNumConformers() for m in preloaded_mols]
        median_confs = sorted(actual_confs_all)[len(actual_confs_all) // 2]
        conformer_counts = [median_confs]
        print(f"  Using {len(preloaded_mols)} preloaded molecules (~{median_confs} conformers each)")
    elif conformer_counts is None:
        conformer_counts = [5, 10, 20]

    results = []

    print("=" * 70)
    print("TFD Benchmark: RDKit vs nvMolKit (CPU) vs nvMolKit (GPU)")
    print(f"Molecule counts: {mol_counts}")
    print(f"Conformer counts: {conformer_counts}")
    print("=" * 70)

    for num_confs in conformer_counts:
        if preloaded_mols is not None:
            all_mols = preloaded_mols[: max(mol_counts)]
        else:
            print(f"\n--- Preparing molecules with {num_confs} conformers ---")
            all_mols = prepare_molecules(
                smiles_list, num_confs, max_mols=max(mol_counts) + 20, smiles_file=smiles_file
            )

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

            # nvMolKit CPU list benchmark (return_type="list")
            try:
                t, s = time_it(lambda: bench_nvmol_cpu_list(mols))
                result["nvmol_cpu_list_time_ms"] = t
                result["nvmol_cpu_list_std_ms"] = s
                print(f"  nvMolKit (CPU list): {t:8.2f} ms (+/- {s:.2f})")
            except Exception as e:
                print(f"  nvMolKit CPU list failed: {e}")
                result["nvmol_cpu_list_time_ms"] = None
                result["nvmol_cpu_list_std_ms"] = None

            # nvMolKit CPU numpy benchmark (return_type="numpy")
            try:
                t, s = time_it(lambda: bench_nvmol_cpu_numpy(mols))
                result["nvmol_cpu_numpy_time_ms"] = t
                result["nvmol_cpu_numpy_std_ms"] = s
                print(f"  nvMolKit (CPU numpy):{t:8.2f} ms (+/- {s:.2f})")
            except Exception as e:
                print(f"  nvMolKit CPU numpy failed: {e}")
                result["nvmol_cpu_numpy_time_ms"] = None
                result["nvmol_cpu_numpy_std_ms"] = None

            # nvMolKit GPU list benchmark (return_type="list")
            try:
                t, s = time_it(lambda: bench_nvmol_gpu_list(mols))
                result["nvmol_gpu_list_time_ms"] = t
                result["nvmol_gpu_list_std_ms"] = s
                print(f"  nvMolKit (GPU list):  {t:8.2f} ms (+/- {s:.2f})")
            except Exception as e:
                print(f"  nvMolKit GPU list failed: {e}")
                result["nvmol_gpu_list_time_ms"] = None

            # nvMolKit GPU numpy benchmark (return_type="numpy")
            try:
                t, s = time_it(lambda: bench_nvmol_gpu_numpy(mols))
                result["nvmol_gpu_numpy_time_ms"] = t
                result["nvmol_gpu_numpy_std_ms"] = s
                print(f"  nvMolKit (GPU numpy): {t:8.2f} ms (+/- {s:.2f})")
            except Exception as e:
                print(f"  nvMolKit GPU numpy failed: {e}")
                result["nvmol_gpu_numpy_time_ms"] = None

            # nvMolKit GPU tensor benchmark (return_type="tensor", no D2H)
            try:
                t, s = time_it(lambda: bench_nvmol_gpu_tensor(mols))
                result["nvmol_gpu_tensor_time_ms"] = t
                result["nvmol_gpu_tensor_std_ms"] = s
                print(f"  nvMolKit (GPU ten):  {t:8.2f} ms (+/- {s:.2f})")
            except Exception as e:
                print(f"  nvMolKit GPU tensor failed: {e}")
                result["nvmol_gpu_tensor_time_ms"] = None

            # Calculate speedups vs RDKit
            speedups = {}
            for key, label in [
                ("nvmol_cpu_list_time_ms", "CPU list"),
                ("nvmol_cpu_numpy_time_ms", "CPU numpy"),
                ("nvmol_gpu_list_time_ms", "GPU list"),
                ("nvmol_gpu_numpy_time_ms", "GPU numpy"),
                ("nvmol_gpu_tensor_time_ms", "GPU tensor"),
            ]:
                if result.get("rdkit_time_ms") and result.get(key):
                    speedups[label] = result["rdkit_time_ms"] / result[key]

            for label, val in speedups.items():
                print(f"  Speedup {label:>10s} vs RDKit: {val:.1f}x")

            results.append(result)

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")

    return df


def main():
    _default_smiles = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "benchmark_smiles.csv")

    parser = argparse.ArgumentParser(description="TFD Benchmark")
    parser.add_argument(
        "--smiles-file",
        type=str,
        default=_default_smiles,
        help="CSV file with SMILES (default: benchmarks/data/benchmark_smiles.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tfd_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--num-mols",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 500, 1000],
        help="Molecule counts to benchmark (default: 1 10 50 100 500 1000)",
    )
    parser.add_argument(
        "--num-confs",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Conformer counts to benchmark (default: 5 10 20 50)",
    )
    parser.add_argument(
        "--skip-rdkit",
        action="store_true",
        help="Skip RDKit benchmarks (faster)",
    )
    parser.add_argument(
        "--pkl-file",
        type=str,
        nargs="+",
        default=None,
        help="Precomputed pickle file(s) containing molecules with conformers. "
        "When provided, --smiles-file and --num-confs are ignored.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify correctness before benchmarking",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify correctness and exit (skip benchmarking)",
    )
    args = parser.parse_args()

    preloaded_mols = None
    smiles_list = None

    if args.pkl_file:
        print("Loading precomputed molecules from pickle file(s)...")
        preloaded_mols = load_pkl_files(args.pkl_file)
        if not preloaded_mols:
            print("Error: no molecules loaded from pickle files")
            sys.exit(1)
        print(f"Total: {len(preloaded_mols)} molecules")
    else:
        print(f"Loading SMILES from: {args.smiles_file}")
        try:
            df = pd.read_csv(args.smiles_file)
            smiles_list = df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error loading SMILES file: {e}")
            sys.exit(1)
        print(f"Loaded {len(smiles_list)} SMILES")

    if args.verify or args.verify_only:
        print("\nVerifying correctness...")
        if preloaded_mols is not None:
            test_mols = preloaded_mols[:50]
        else:
            test_mols = prepare_molecules(smiles_list, num_confs=5, max_mols=50, smiles_file=args.smiles_file)
        all_correct = True
        mismatches = 0
        for i, mol in enumerate(test_mols):
            if verify_correctness(mol):
                print(f"  Molecule {i}: OK")
            else:
                print(f"  Molecule {i}: MISMATCH")
                all_correct = False
                mismatches += 1
        if all_correct:
            print(f"All {len(test_mols)} molecules match RDKit.")
        else:
            print(f"Warning: {mismatches}/{len(test_mols)} molecules did not match RDKit within tolerance")

        if args.verify_only:
            sys.exit(0 if all_correct else 1)

    run_benchmarks(
        smiles_list=smiles_list,
        skip_rdkit=args.skip_rdkit,
        output_file=args.output,
        smiles_file=args.smiles_file,
        mol_counts=args.num_mols,
        conformer_counts=args.num_confs if not args.pkl_file else None,
        preloaded_mols=preloaded_mols,
    )


if __name__ == "__main__":
    main()
